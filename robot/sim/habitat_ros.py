"""Sprocket's hardware in a Habitat home, on ROS: what the real robot's drivers publish, made from the simulated world.

The real robot has camera drivers (three colour cameras), an MCU (the motor controller that takes /cmd_vel, wheel
odometry, the ToF grids) and an IMU. This node stands in for those and nothing else: the robot's own software (the
EKF, SLAM or AMCL, Nav2, the depth node, perception, the agent) runs on top unchanged and never sees the truth.

It owns the clock. Simulated time runs at `rtf` times wall-clock time, so that software running slower here than on
the board still gets about the board's share of work done per simulated second. Errors are modelled where the real
parts have them: the wheels misjudge distance a little and slip when turning on the spot, the gyro has a bias, the ToF
zones are noisy. The truth goes out on /ground_truth for the judge only.

    python -m robot.sim.habitat_ros --port <the Habitat server's> --rtf 0.5
"""

from __future__ import annotations

import argparse
import array
import math
import random
import time
from typing import Any

import numpy as np

from robot.ros.model import CAMERAS, MAX_DV, MAX_DW, MAX_V, MAX_W, TOF, TRACK, WHEEL_R
from robot.sim.habitat import Habitat

DT = 0.02  # the motor controller's period, simulated seconds
WATCHDOG = 0.3  # the MCU stops the wheels when no command came for this long
TOF_EVERY = 5  # ticks: 10 Hz
TRUTH_EVERY = 5
# what the real parts get wrong (one draw per run for the biases; noise per reading)
WHEEL_SCALE = 0.01  # wheel odometry's distance error, as a fraction (tyre wear, a calibration a little off)
TURN_SLIP = 0.05  # wheels slip when the body turns on the spot: they report this much more turning than happened
WHEEL_NOISE_V, WHEEL_NOISE_W = 0.005, 0.02  # m/s, rad/s per reading
GYRO_BIAS, GYRO_NOISE = 0.003, 0.005  # rad/s
TOF_NOISE = (0.005, 0.01)  # metres: fixed + per metre of range


def _approach(now: float, want: float, step: float) -> float:
    return now + max(-step, min(step, want - now))


class Hardware:
    def __init__(self, port: int, rtf: float, seed: int = 0) -> None:
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
        from std_msgs.msg import Float32MultiArray, Int32

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("sprocket_hardware")
        self.msgs = {"clock": Clock, "odom": Odometry, "imu": Imu, "joints": JointState, "image": Image,
                     "info": CameraInfo, "tof": Float32MultiArray, "count": Int32}
        self.world = Habitat(port)
        self.rtf = rtf
        rng = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.wheel_scale = 1 + rng.gauss(0, WHEEL_SCALE)
        self.gyro_bias = rng.gauss(0, GYRO_BIAS)
        start = self.world.call(op="info")["start"]
        self.truth = self.world.call(op="reset", pose=start)
        self.t = 0.0
        self.v = self.w = 0.0
        self.cmd, self.cmd_t = (0.0, 0.0), -1.0
        self.odom = [0.0, 0.0, 0.0]  # what the wheels make of the way so far
        self.wheels = [0.0, 0.0]  # wheel angles, radians
        self.bumps = 0
        self.pressing = False
        pub = self.node.create_publisher
        self.pubs = {
            "clock": pub(Clock, "/clock", 10), "odom": pub(Odometry, "/odom", 10), "imu": pub(Imu, "/imu", 10),
            "joints": pub(JointState, "/joint_states", 10), "truth": pub(Odometry, "/ground_truth", 10),
            "bumps": pub(Int32, "/ground_truth/bumps", 10),
            **{f"tof_{side}": pub(Float32MultiArray, f"/tof/{side}", 10) for side in TOF},
            **{f"{name}/image": pub(Image, f"/{name}/image", 2) for name in CAMERAS},
            **{f"{name}/info": pub(CameraInfo, f"/{name}/camera_info", 2) for name in CAMERAS},
        }
        self.node.create_subscription(TwistStamped, "/cmd_vel", self._command, 10)
        # the simulator's own control, not the robot's: put the body back on its dock between eval tasks
        from std_msgs.msg import Empty

        self.start, self.reset_asked = start, False
        self.node.create_subscription(Empty, "/sim/reset", lambda _: setattr(self, "reset_asked", True), 1)
        ticks = {name: max(1, round(1 / (rate * DT))) for name, (*_, rate) in CAMERAS.items()}
        self.cameras = [(name, ticks[name], k * ticks[name] // len(CAMERAS)) for k, name in enumerate(CAMERAS)]
        self.rendered: list[float] = []  # wall seconds per picture, for the report

    def _command(self, msg: Any) -> None:
        self.cmd, self.cmd_t = (msg.twist.linear.x, msg.twist.angular.z), self.t

    def _stamp(self) -> Any:
        """The time as a ROS stamp, from whole nanoseconds (splitting the float seconds instead once turned 4.99999
        into 4 s + 1e9 ns, wrapped to 4.0: the clock stepped back each second and every node cleared its TF)."""
        from builtin_interfaces.msg import Time

        sec, nanosec = divmod(round(self.t * 1e9), 1_000_000_000)
        return Time(sec=sec, nanosec=nanosec)

    # ---- one motor-controller period ---------------------------------------------------------------------------
    def tick(self, k: int) -> None:
        if self.reset_asked:  # the wheels' own count carries on: the robot was carried, not driven
            self.reset_asked, self.v, self.w, self.cmd = False, 0.0, 0.0, (0.0, 0.0)
            self.truth = self.world.call(op="reset", pose=self.start)
        want = self.cmd if self.t - self.cmd_t <= WATCHDOG else (0.0, 0.0)
        self.v = _approach(self.v, max(-MAX_V, min(MAX_V, want[0])), MAX_DV * DT)
        self.w = _approach(self.w, max(-MAX_W, min(MAX_W, want[1])), MAX_DW * DT)
        before = self.truth
        step = self.world.call(op="step", v=self.v, w=self.w, dt=DT)
        self.truth = step["pose"]
        if step["bumped"]:
            if not self.pressing:  # a new contact: where, for the judge's log
                x, y, yaw = self.truth
                print(f"bump at ({x:.2f}, {y:.2f}) heading {math.degrees(yaw):.0f} deg, t {self.t:.1f} s", flush=True)
            self.bumps += 1
            self.v = 0.0  # pressed against something: the wheels stall
        self.pressing = step["bumped"]
        self.t += DT
        stamp = self._stamp()
        clock = self.msgs["clock"]()
        clock.clock = stamp
        self.pubs["clock"].publish(clock)
        moved = math.dist(before[:2], self.truth[:2])
        turned = math.atan2(math.sin(self.truth[2] - before[2]), math.cos(self.truth[2] - before[2]))
        v_true, w_true = math.copysign(moved, self.v) / DT, turned / DT
        self._odometry(stamp, v_true, w_true)
        self._imu(stamp, w_true)
        if k % TOF_EVERY == 0:
            self._tof()
        for name, every, offset in self.cameras:
            if k % every == offset:
                self._camera(name, stamp)
        if k % TRUTH_EVERY == 0:
            self._truth(stamp)

    def _odometry(self, stamp: Any, v_true: float, w_true: float) -> None:
        """What the wheel encoders report: the wheels turn by what the body did, plus their own errors."""
        spin = abs(w_true) > 0.2 and abs(v_true) < 0.05  # turning on the spot: the tyres scrub
        v = v_true * self.wheel_scale + self.rng.normal(0, WHEEL_NOISE_V) * (v_true != 0 or w_true != 0)
        w = w_true * (1 + TURN_SLIP if spin else 1) + self.rng.normal(0, WHEEL_NOISE_W) * (w_true != 0)
        x, y, yaw = self.odom
        self.odom = [x + v * DT * math.cos(yaw), y + v * DT * math.sin(yaw), yaw + w * DT]
        for i, side in enumerate((-1, 1)):  # left, right
            self.wheels[i] += (v + side * w * TRACK / 2) / WHEEL_R * DT
        msg = self.msgs["odom"]()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, "odom", "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.odom[0], self.odom[1]
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(self.odom[2] / 2), math.cos(self.odom[2] / 2)
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = v, w
        cov = [0.0] * 36
        cov[0], cov[7], cov[35] = 0.0004, 1e-6, 0.004  # vx, vy (held at zero), vyaw
        msg.twist.covariance = cov
        self.pubs["odom"].publish(msg)

    def _imu(self, stamp: Any, w_true: float) -> None:
        msg = self.msgs["imu"]()
        msg.header.stamp, msg.header.frame_id = stamp, "imu_link"
        yaw = self.truth[2]  # level floor: no roll or pitch (the robot fuses only the turn rate)
        msg.orientation.z, msg.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        msg.angular_velocity.z = w_true + self.gyro_bias + self.rng.normal(0, GYRO_NOISE)
        msg.linear_acceleration.z = 9.81
        msg.orientation_covariance[0] = msg.orientation_covariance[4] = msg.orientation_covariance[8] = 0.01
        msg.angular_velocity_covariance[8] = GYRO_NOISE ** 2
        msg.linear_acceleration_covariance[0] = msg.linear_acceleration_covariance[4] = 0.01
        msg.linear_acceleration_covariance[8] = 0.01
        self.pubs["imu"].publish(msg)
        joints = self.msgs["joints"]()
        joints.header.stamp = stamp
        joints.name, joints.position = ["left_wheel_joint", "right_wheel_joint"], list(self.wheels)
        self.pubs["joints"].publish(joints)

    def _tof(self) -> None:
        grids = self.world.call(op="tof")
        for side, raw in grids.items():
            grid = np.frombuffer(raw, np.float32).copy()
            ok = np.isfinite(grid)
            grid[ok] += self.rng.normal(0, 1, ok.sum()) * (TOF_NOISE[0] + TOF_NOISE[1] * grid[ok])
            msg = self.msgs["tof"]()
            msg.data = array.array("f", grid.tobytes())
            self.pubs[f"tof_{side}"].publish(msg)

    def _camera(self, name: str, stamp: Any) -> None:
        started = time.monotonic()
        _, hfov, width, height, _ = CAMERAS[name]
        data = self.world.call(op="camera", name=name)
        msg = self.msgs["image"]()
        msg.header.stamp, msg.header.frame_id = stamp, f"{name}_optical_frame"
        msg.height, msg.width, msg.encoding, msg.step = height, width, "rgb8", width * 3
        msg.data = array.array("B", data)  # the fast path: no per-byte checks
        self.pubs[f"{name}/image"].publish(msg)
        info = self.msgs["info"]()
        info.header = msg.header
        info.height, info.width, info.distortion_model = height, width, "plumb_bob"
        f = (width / 2) / math.tan(hfov / 2)
        info.k = [f, 0.0, width / 2, 0.0, f, height / 2, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, width / 2, 0.0, 0.0, f, height / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = [0.0] * 5
        self.pubs[f"{name}/info"].publish(info)
        self.rendered.append(time.monotonic() - started)

    def _truth(self, stamp: Any) -> None:
        x, y, yaw = self.truth
        msg = self.msgs["odom"]()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, "world", "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pubs["truth"].publish(msg)
        count = self.msgs["count"]()
        count.data = self.bumps
        self.pubs["bumps"].publish(count)

    # ---- the clock ---------------------------------------------------------------------------------------------
    def run(self) -> None:
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()
        wall0, k, reported = time.monotonic(), 0, time.monotonic()
        while rclpy.ok():
            k += 1
            self.tick(k)
            ahead = wall0 + self.t / self.rtf - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
            if time.monotonic() - reported > 30:
                reported = time.monotonic()
                pics = sorted(self.rendered[-50:])
                print(f"hardware: simulated {self.t:.0f} s at {self.t / (reported - wall0):.2f} x wall time "
                      f"(asked {self.rtf}); {self.bumps} bumps; picture {1000 * pics[len(pics) // 2]:.0f} ms",
                      flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="the Habitat server's port (robot/sim/habitat.py)")
    parser.add_argument("--rtf", type=float, default=0.5, help="simulated seconds per wall-clock second")
    parser.add_argument("--seed", type=int, default=0, help="for the sensor errors")
    args = parser.parse_args()
    Hardware(args.port, args.rtf, args.seed).run()


if __name__ == "__main__":
    main()
