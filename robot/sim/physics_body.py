"""Rigid-body MuJoCo simulation of companion_r1 with motor/servo actuators, odometry and sensor noise.

Commands drive the wheel velocity and head servo actuators of `config/companion_r1_physics.xml`; a 100 Hz
"firmware" loop inside the 2 ms physics step closes the speed loop and integrates odometry (quantised wheel
encoders + noisy IMU yaw), as the MCU does. `pose()` is that estimate; `true_pose()` is MuJoCo ground truth.
Time only moves in `advance(seconds)`. Chassis contact or stalled wheels raise a "bumper" reflex. Movable
sandbox objects are free bodies the robot can push; their new positions go back with `sandbox.move_object()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot.body.hal import BodyCapabilities, BodyPose, Reflex, SensorReadings, load_capabilities
from robot.core.angles import wrap_deg
from robot.core.config import PROJECT_ROOT

PHYSICS_XML_PATH = PROJECT_ROOT / "config" / "companion_r1_physics.xml"
_WHEELS = ("fl", "fr", "rl", "rr")
_RAY_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # group 3 = the robot's own geoms
_TOF_FAN_DEG = (-12.0, -6.0, 0.0, 6.0, 12.0)
_TOF_MAX_MM = 2500.0
_PARK_XY_M = (4.0, 4.0)  # outside the walls; the floor plane is infinite


@dataclass(frozen=True)
class SensorNoise:
    """1-sigma sensor errors. Systematic terms are drawn once per body from its seed."""

    wheel_scale: float = 0.01  # relative wheel-radius calibration error, per side
    gyro_bias_deg_s: float = 0.05
    gyro_noise_deg_s: float = 0.2  # white noise per 100 Hz IMU sample
    tof_mm: float = 3.0

    @classmethod
    def off(cls) -> SensorNoise:
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass
class _PhysObject:
    """Physics body mirroring one SimulationSandbox object."""

    body: str
    free: bool
    rest_z: float
    dynamic: bool
    signature: tuple | None = None
    in_use: bool = False
    sandbox_xy: tuple[float, float] | None = None
    physics_xy: tuple[float, float] | None = None
    qadr: int = -1
    vadr: int = -1
    mocap: int = -1


def _signature(obj: dict[str, Any]) -> tuple:
    return (obj["shape"], obj["width_mm"], obj["depth_mm"], obj["height_mm"], bool(obj.get("movable", True)))


class PhysicsBody:
    """MuJoCo rigid-body companion_r1 with skid-steer wheel motors, head servo, odometry and bumper."""

    VELOCITY_TIMEOUT_S = 0.35  # motor watchdog: no fresh speed command -> stop
    LINEAR_ACCEL_MM_S2 = 800.0
    ANGULAR_ACCEL_DEG_S2 = 360.0
    RATE_P = 0.5
    RATE_I = 3.0
    TURN_SLIP_FACTOR = 2.4  # effective/nominal track width of the skid-steer base (feed-forward only)
    BUMPER_FORCE_N = 6.0
    BUMPER_HOLD_S = 0.04
    STALL_HOLD_S = 0.3
    FIRMWARE_DT_S = 0.01
    OBJECT_SYNC_MM = 5.0

    def __init__(self, capabilities: BodyCapabilities | None = None, *, sandbox: Any | None = None, seed: int = 0,
                 battery_idle_pct_per_min: float = 0.5, battery_pct_per_m: float = 1.0,
                 charge_pct_per_min: float = 20.0, noise: SensorNoise | None = None,
                 xml_path: Path | str | None = None) -> None:
        self._caps = capabilities or load_capabilities()
        self._sensors = SensorReadings(front_tof_mm=self._caps.default_front_tof_mm)
        self._world: Any | None = None
        self._vel_target: tuple[float, float] = (0.0, 0.0)
        self._vel_until = 0.0
        self.battery_idle_pct_per_min = float(battery_idle_pct_per_min)
        self.battery_pct_per_m = float(battery_pct_per_m)
        self.charge_pct_per_min = float(charge_pct_per_min)
        self.noise = noise or SensorNoise()
        self._rng = np.random.default_rng(seed)
        self._wheel_scale = tuple(1.0 + self._rng.normal(0.0, self.noise.wheel_scale) for _ in range(2))
        self._gyro_bias = float(self._rng.normal(0.0, self.noise.gyro_bias_deg_s))
        self._r_nominal_m = self._caps.wheel_diameter_mm / 2000.0
        self._half_track_m = self._caps.track_width_mm * self.TURN_SLIP_FACTOR / 2000.0

        # MuJoCo cannot open non-ASCII paths on Windows, so always parse from a string.
        xml = Path(xml_path or PHYSICS_XML_PATH).read_text(encoding="utf-8")
        self._spec = mujoco.MjSpec.from_string(xml)
        self._model = self._spec.compile()
        self._data = mujoco.MjData(self._model)
        self._xml_objects: dict[str, _PhysObject] = {}
        self._dyn_objects: dict[str, _PhysObject] = {}
        self._dyn_serial = 0
        for i in range(self._model.nbody):
            body = self._model.body(i)
            if body.name.startswith("obj_"):
                free = int(self._model.body_jntnum[i]) > 0
                z = float(self._model.body_pos[i][2])
                self._xml_objects[body.name[4:]] = _PhysObject(body.name, free, z, dynamic=False, in_use=True)
        self._index_model()
        self._fw_every = max(1, round(self.FIRMWARE_DT_S / self._model.opt.timestep))
        self._contact_force = np.zeros(6)
        self._ray_geom = np.zeros(1, dtype=np.int32)

        self._fw_v = 0.0  # speed-control set point ramped by the firmware (mm/s, deg/s)
        self._fw_w = 0.0
        self._fw_w_i = 0.0
        self._fw_blocked = False
        self._fw_reflex: Reflex | None = None
        self._head_setpoint = 0.0
        self._head_target = 0.0
        self._head_speed = self._caps.head_tilt.max_speed_deg_s
        self._bump_s = 0.0
        self._stall_s = 0.0
        self._phase = 0
        self._time_debt = 0.0
        self._sim_time = 0.0
        self._awake_until = 0.0
        self._battery_travel_mm = 0.0

        # Let the tyres settle onto the floor before odometry is zeroed.
        mujoco.mj_step(self._model, self._data, nstep=round(0.3 / self._model.opt.timestep))
        self._reset_odometry()
        self._battery_ref = 0.0
        if sandbox is not None:
            self.bind_world(sandbox)

    # -- odometry estimate (what pose() reports) ------------------------------------------------

    def capabilities(self) -> BodyCapabilities:
        return self._caps

    def pose(self) -> BodyPose:
        return self._odom.copy()

    def sensors(self) -> SensorReadings:
        self._refresh_sensors()
        return self._sensors.copy()
    def reset_pose(self, pose: BodyPose | None = None) -> None:
        """Teleport the physical robot (and head) to `pose`, at rest, and re-zero odometry there.

        """
        pose = pose or BodyPose()
        m, d = self._model, self._data
        a, v = self._base_qadr, self._base_vadr
        yaw = math.radians(pose.heading_deg)
        d.qpos[a : a + 3] = (pose.x_mm / 1000.0, pose.y_mm / 1000.0, self._base_rest_z)
        d.qpos[a + 3 : a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[v : v + 6] = 0.0
        d.qvel[self._wheel_vadr] = 0.0
        tilt = self._caps.clamp_head(pose.head_tilt_deg)
        d.qpos[self._head_qadr] = math.radians(tilt)
        d.qvel[self._head_vadr] = 0.0
        self._halt_actuators()
        self._head_setpoint = self._head_target = tilt
        d.ctrl[self._servo_id] = math.radians(tilt)
        mujoco.mj_forward(m, d)
        self._reset_odometry()
        self._sync_world()
        self._wake()

    def _reset_odometry(self) -> None:
        truth = self.true_pose()
        self._odom = BodyPose(
            x_mm=truth.x_mm, y_mm=truth.y_mm, heading_deg=truth.heading_deg, head_tilt_deg=self._head_setpoint
        )
        self._yaw_est = self._imu_yaw_ref = truth.heading_deg
        self._enc_ticks = self._encoder_ticks()
        self._sensors.wheel_ticks_left, self._sensors.wheel_ticks_right = self._enc_ticks
        self._sensors.imu_yaw_deg = truth.heading_deg
        self._sensors.imu_pitch_deg = self._head_setpoint

    def _true_yaw_deg(self) -> float:
        a = self._base_qadr
        w, x, y, z = (float(c) for c in self._data.qpos[a + 3 : a + 7])
        return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def true_pose(self) -> BodyPose:
        """Ground-truth pose from the physics state (never available to the robot itself)."""
        q, qv = self._data.qpos, self._data.qvel
        a, v = self._base_qadr, self._base_vadr
        yaw = math.radians(self._true_yaw_deg())
        forward = float(qv[v]) * math.cos(yaw) + float(qv[v + 1]) * math.sin(yaw)
        return BodyPose(
            x_mm=float(q[a]) * 1000.0,
            y_mm=float(q[a + 1]) * 1000.0,
            heading_deg=wrap_deg(math.degrees(yaw)),
            head_tilt_deg=math.degrees(float(q[self._head_qadr])),
            linear_vel_mm_s=forward * 1000.0,
            angular_vel_deg_s=math.degrees(float(qv[v + 5])),
        )

    # -- commands and time -----------------------------------------------------------------------

    def command(self, linear_mm_s: float, angular_deg_s: float) -> None:
        """Drive at this speed until the next command; the watchdog stops it if none comes."""
        v, w = self._caps.clamp_velocity(linear_mm_s, angular_deg_s)
        self._vel_target = (0.0, 0.0) if self._sensors.estop_active else (v, w)
        self._vel_until = self._sim_time + self.VELOCITY_TIMEOUT_S

    def set_head(self, tilt_deg: float) -> None:
        self._head_target = self._caps.clamp_head(tilt_deg)
        self._wake()

    def halt(self) -> None:
        self._halt_actuators()

    def set_estop(self, active: bool) -> None:
        self._sensors.estop_active = active
        if active:
            self._halt_actuators()

    def bind_world(self, sandbox: Any) -> None:
        """Sample its range sensors from ground truth, mirror its objects into physics, report pushed ones back."""
        self._world = sandbox
        self._sync_world()

    def advance(self, seconds: float) -> Reflex | None:
        """Run the world forward; returns the reflex that limited motion, if any."""
        self._time_debt += max(0.0, seconds)
        dt = self._model.opt.timestep
        steps = int(self._time_debt / dt + 1e-9)
        self._time_debt -= steps * dt
        if steps > 0:
            if self._at_rest():
                self._sim_time += steps * dt
            else:
                self._step_physics(steps)
        self._sync_world()
        self._refresh_sensors()
        reflex: Reflex | None = None
        if self._fw_reflex is not None:
            reflex, self._fw_reflex = self._fw_reflex, None
        elif self._sensors.estop_active:
            reflex = Reflex.ESTOP
        elif self._sensors.cliff_triggered or self._sensors.picked_up:
            reflex = Reflex.CLIFF
        elif self._fw_blocked:
            reflex = Reflex.OBSTACLE_AHEAD
        if reflex is not None and reflex.halts:
            self._halt_actuators()
        return reflex

    def _refresh_sensors(self) -> None:
        truth = self.true_pose()
        if self._world is not None:
            tof = float(self._world.compute_front_tof_mm(truth))
            cliff = bool(self._world.check_cliff_triggered(truth))
        else:
            tof, cliff = self._raycast_tof_mm(), False
        if self.noise.tof_mm > 0:
            tof = max(0.0, tof + float(self._rng.normal(0.0, self.noise.tof_mm)))
        self._sensors.front_tof_mm, self._sensors.cliff_triggered = round(tof, 1), cliff
        self._update_battery(truth)

    def _update_battery(self, truth: BodyPose) -> None:
        """Idle drain per minute plus drain per metre of wheel travel; charging while docked."""
        dt, self._battery_ref = max(0.0, self._sim_time - self._battery_ref), self._sim_time
        travel_m, self._battery_travel_mm = self._battery_travel_mm / 1000.0, 0.0
        charging = self._world is not None and self._world.is_charging(truth)
        pct = self._sensors.battery_pct - self.battery_pct_per_m * travel_m
        if charging:
            pct += self.charge_pct_per_min * dt / 60.0
        else:
            pct -= self.battery_idle_pct_per_min * dt / 60.0
        self._sensors.battery_pct = min(100.0, max(0.0, pct))
        self._sensors.charging = charging

    def _halt_actuators(self) -> None:
        self._vel_target, self._vel_until = (0.0, 0.0), self._sim_time
        self._fw_v = self._fw_w = self._fw_w_i = 0.0
        self._head_target = self._head_setpoint
        self._data.ctrl[self._motor_ids] = 0.0

    def _wake(self, seconds: float = 1.0) -> None:
        self._awake_until = max(self._awake_until, self._sim_time + seconds)

    def _at_rest(self) -> bool:
        return (
            not self._velocity_active()
            and self._head_setpoint == self._head_target
            and self._sim_time >= self._awake_until
            and float(np.max(np.abs(self._data.qvel))) < 1e-3
        )

    def _step_physics(self, steps: int) -> None:
        m, d = self._model, self._data
        while steps > 0:
            chunk = min(steps, self._fw_every - self._phase)
            mujoco.mj_step(m, d, nstep=chunk)
            steps -= chunk
            self._phase += chunk
            self._sim_time += chunk * m.opt.timestep
            if self._phase >= self._fw_every:
                self._phase = 0
                self._firmware_tick(self._fw_every * m.opt.timestep)

    # -- 100 Hz firmware: odometry, motion control, head servo, bumper -----------------------------

    def _encoder_ticks(self) -> tuple[int, int]:
        q = self._data.qpos
        fl, fr, rl, rr = (float(q[i]) for i in self._wheel_qadr)
        per_rad = self._caps.encoder_ticks_per_rev / (2.0 * math.pi)
        return round(0.5 * (fl + rl) * per_rad), round(0.5 * (fr + rr) * per_rad)

    def _firmware_tick(self, dt: float) -> None:
        d = self._data
        ticks = self._encoder_ticks()
        mm_per_tick = math.pi * self._caps.wheel_diameter_mm / self._caps.encoder_ticks_per_rev
        dl = (ticks[0] - self._enc_ticks[0]) * mm_per_tick * self._wheel_scale[0]
        dr = (ticks[1] - self._enc_ticks[1]) * mm_per_tick * self._wheel_scale[1]
        self._enc_ticks = ticks
        self._battery_travel_mm += 0.5 * (abs(dl) + abs(dr))
        # IMU reports the mean rate over the sample (internally integrated at a high rate).
        yaw = self._true_yaw_deg()
        gyro = wrap_deg(yaw - self._imu_yaw_ref) / dt + self._gyro_bias
        self._imu_yaw_ref = yaw
        if self.noise.gyro_noise_deg_s > 0:
            gyro += float(self._rng.normal(0.0, self.noise.gyro_noise_deg_s))
        # Zero-velocity gating: a stationary robot does not integrate gyro bias into its heading.
        if dl or dr or abs(gyro) > 1.0:
            dyaw = gyro * dt
        else:
            dyaw = gyro = 0.0
        ds = 0.5 * (dl + dr)
        mid = math.radians(self._yaw_est + 0.5 * dyaw)
        self._yaw_est += dyaw
        odom = self._odom
        odom.x_mm += ds * math.cos(mid)
        odom.y_mm += ds * math.sin(mid)
        odom.heading_deg = wrap_deg(self._yaw_est)
        odom.linear_vel_mm_s += 0.3 * (ds / dt - odom.linear_vel_mm_s)
        odom.angular_vel_deg_s += 0.3 * (gyro - odom.angular_vel_deg_s)
        self._sensors.wheel_ticks_left, self._sensors.wheel_ticks_right = ticks
        self._sensors.imu_yaw_deg = odom.heading_deg

        v_left, v_right = self._velocity_control(dt, gyro)
        d.ctrl[self._left_motors] = v_left / self._r_nominal_m
        d.ctrl[self._right_motors] = v_right / self._r_nominal_m
        self._servo_tick(dt)
        self._monitor_bumper(dt, v_left, v_right)

    def _velocity_active(self) -> bool:
        return self._sim_time < self._vel_until or abs(self._fw_v) > 1e-3 or abs(self._fw_w) > 1e-3

    def _velocity_control(self, dt: float, gyro_deg_s: float) -> tuple[float, float]:
        """Speed control: ramp to the commanded speeds, hold the turn rate on the gyro (watchdog included)."""
        fresh = self._sim_time < self._vel_until
        target_v, target_w = self._vel_target if fresh else (0.0, 0.0)
        tof = self._sensors.front_tof_mm
        self._fw_blocked = target_v > 0 and (not math.isfinite(tof) or tof <= self._caps.min_front_tof_stop_mm)
        if self._fw_blocked:
            target_v = 0.0
        step_v, step_w = self.LINEAR_ACCEL_MM_S2 * dt, self.ANGULAR_ACCEL_DEG_S2 * dt
        self._fw_v += max(-step_v, min(step_v, target_v - self._fw_v))
        self._fw_w += max(-step_w, min(step_w, target_w - self._fw_w))
        if abs(self._fw_v) < 1e-3 and abs(self._fw_w) < 1e-3 and not fresh:
            self._fw_v = self._fw_w = self._fw_w_i = 0.0
            return 0.0, 0.0
        error = self._fw_w - gyro_deg_s
        self._fw_w_i = max(-30.0, min(30.0, self._fw_w_i + error * dt))
        omega = self._fw_w + self.RATE_P * error + self.RATE_I * self._fw_w_i
        turn = math.radians(omega) * self._half_track_m
        return self._fw_v / 1000.0 - turn, self._fw_v / 1000.0 + turn

    def _servo_tick(self, dt: float) -> None:
        step = self._head_speed * dt
        delta = self._head_target - self._head_setpoint
        if abs(delta) <= step:
            self._head_setpoint = self._head_target
        else:
            self._head_setpoint += math.copysign(step, delta)
        self._data.ctrl[self._servo_id] = math.radians(self._head_setpoint)
        # A hobby servo has no position feedback: the estimate is the commanded angle.
        self._odom.head_tilt_deg = self._head_setpoint
        self._sensors.imu_pitch_deg = self._head_setpoint

    def _monitor_bumper(self, dt: float, v_left: float, v_right: float) -> None:
        """Latch a bumper reflex on sustained chassis contact force or on stalled drive wheels."""
        if not self._velocity_active():
            self._bump_s = self._stall_s = 0.0
            return
        force = self._external_contact_force_n()
        self._bump_s = self._bump_s + dt if force >= self.BUMPER_FORCE_N else 0.0
        qv = self._data.qvel
        fl, fr, rl, rr = (float(qv[i]) * self._r_nominal_m for i in self._wheel_vadr)
        stalled = any(
            abs(cmd) > 0.04 and abs(meas) < 0.25 * abs(cmd)
            for cmd, meas in ((v_left, 0.5 * (fl + rl)), (v_right, 0.5 * (fr + rr)))
        )
        self._stall_s = self._stall_s + dt if stalled else 0.0
        if self._bump_s >= self.BUMPER_HOLD_S:
            self._fw_reflex = Reflex.BUMPER
        elif self._stall_s >= self.STALL_HOLD_S:
            self._fw_reflex = Reflex.BUMPER
        else:
            return
        self._halt_actuators()

    def _external_contact_force_n(self) -> float:
        d = self._data
        if d.ncon == 0:
            return 0.0
        geoms = d.contact.geom[: d.ncon]
        total = 0.0
        for i in np.nonzero((geoms == self._chassis_gid).any(axis=1))[0]:
            g1, g2 = geoms[i]
            if self._floor_gid in (g1, g2):
                continue
            mujoco.mj_contactForce(self._model, d, int(i), self._contact_force)
            total += abs(float(self._contact_force[0]))
        return total

    def _raycast_tof_mm(self) -> float:
        """Front ToF from a 5-ray fan (~24 deg field of view) against physics geometry."""
        d = self._data
        yaw = math.radians(self._true_yaw_deg())
        origin = d.site_xpos[self._tof_sid]
        nearest = _TOF_MAX_MM / 1000.0
        for offset in _TOF_FAN_DEG:
            a = yaw + math.radians(offset)
            vec = np.array([math.cos(a), math.sin(a), 0.0])
            dist = mujoco.mj_ray(self._model, d, origin, vec, _RAY_GROUPS, 1, -1, self._ray_geom)
            if 0.0 <= dist < nearest:
                nearest = dist
        return round(max(10.0, min(_TOF_MAX_MM, nearest * 1000.0)), 1)

    # -- sandbox object synchronisation ---------------------------------------------------------

    def _index_model(self) -> None:
        m = self._model
        base = m.joint("base_free")
        self._base_qadr, self._base_vadr = int(base.qposadr[0]), int(base.dofadr[0])
        self._base_rest_z = float(m.qpos0[self._base_qadr + 2])
        wheels = [m.joint(f"{w}_spin") for w in _WHEELS]
        self._wheel_qadr = [int(j.qposadr[0]) for j in wheels]
        self._wheel_vadr = [int(j.dofadr[0]) for j in wheels]
        head = m.joint("head_tilt")
        self._head_qadr, self._head_vadr = int(head.qposadr[0]), int(head.dofadr[0])
        self._motor_ids = [m.actuator(f"motor_{w}").id for w in _WHEELS]
        self._left_motors = [self._motor_ids[0], self._motor_ids[2]]
        self._right_motors = [self._motor_ids[1], self._motor_ids[3]]
        self._servo_id = m.actuator("servo_head").id
        self._chassis_gid = m.geom("chassis").id
        self._floor_gid = m.geom("floor").id
        self._tof_sid = m.site("front_tof").id
        for obj in (*self._xml_objects.values(), *self._dyn_objects.values()):
            body = m.body(obj.body)
            obj.mocap = int(m.body_mocapid[body.id])
            if obj.free:
                jid = int(m.body_jntadr[body.id])
                obj.qadr, obj.vadr = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])

    def _active_objects(self) -> dict[str, _PhysObject]:
        active = {obj_id: obj for obj_id, obj in self._xml_objects.items() if obj.in_use}
        active.update(self._dyn_objects)
        return active

    def _object_xy_mm(self, obj: _PhysObject) -> tuple[float, float]:
        if obj.free:
            q = self._data.qpos
            return float(q[obj.qadr]) * 1000.0, float(q[obj.qadr + 1]) * 1000.0
        pos = self._data.mocap_pos[obj.mocap]
        return float(pos[0]) * 1000.0, float(pos[1]) * 1000.0

    def object_xy_mm(self, obj_id: str) -> tuple[float, float] | None:
        """Ground-truth floor position of a mirrored sandbox object, or None if it is not mirrored."""
        obj = self._active_objects().get(obj_id)
        return None if obj is None else self._object_xy_mm(obj)

    def _place_object(self, obj: _PhysObject, x_m: float, y_m: float) -> None:
        """Teleport an object upright and at rest (sandbox edits are instantaneous)."""
        d = self._data
        if obj.free:
            d.qpos[obj.qadr : obj.qadr + 7] = (x_m, y_m, obj.rest_z, 1.0, 0.0, 0.0, 0.0)
            d.qvel[obj.vadr : obj.vadr + 6] = 0.0
        else:
            d.mocap_pos[obj.mocap] = (x_m, y_m, obj.rest_z)

    def _sync_world(self) -> None:
        """Sandbox edits (add/remove/drag/reset) move physics bodies; pushed bodies move sandbox objects."""
        if self._world is None or self._data is None:
            return
        listing = {o["id"]: o for o in self._world.list_objects()}
        changed = self._reconcile_bodies(listing)
        for obj_id, obj in self._active_objects().items():
            o = listing[obj_id]
            sandbox_xy = (float(o["x_mm"]), float(o["y_mm"]))
            if obj.sandbox_xy is None or math.dist(sandbox_xy, obj.sandbox_xy) > 0.5:
                self._place_object(obj, sandbox_xy[0] / 1000.0, sandbox_xy[1] / 1000.0)
                obj.sandbox_xy = obj.physics_xy = sandbox_xy
                changed = True
            elif obj.free:
                physics_xy = self._object_xy_mm(obj)
                if math.dist(physics_xy, obj.physics_xy) > self.OBJECT_SYNC_MM:
                    moved = self._world.move_object(obj_id, *physics_xy)
                    obj.physics_xy = physics_xy
                    if moved is not None:
                        obj.sandbox_xy = (round(moved.x_mm, 1), round(moved.y_mm, 1))
        if changed:
            mujoco.mj_forward(self._model, self._data)
            self._wake()

    def _reconcile_bodies(self, listing: dict[str, dict[str, Any]]) -> bool:
        """Adopt, park, create or delete physics bodies so they match the sandbox's object list."""
        spec_changed = parked = False
        for obj_id, obj in list(self._dyn_objects.items()):
            o = listing.get(obj_id)
            if o is None or _signature(o) != obj.signature:
                self._spec.delete(self._spec.body(obj.body))
                del self._dyn_objects[obj_id]
                spec_changed = True
        for slot, (obj_id, obj) in enumerate(self._xml_objects.items()):
            o = listing.get(obj_id)
            if o is not None and obj.signature is None:
                obj.signature = _signature(o)  # the XML body stands in for the sandbox object first seen
            wanted = o is not None and obj_id not in self._dyn_objects and _signature(o) == obj.signature
            if wanted != obj.in_use:
                obj.in_use, obj.sandbox_xy = wanted, None
                if not wanted:
                    self._place_object(obj, _PARK_XY_M[0] + 0.5 * slot, _PARK_XY_M[1])
                    parked = True
        for obj_id, o in listing.items():
            xml = self._xml_objects.get(obj_id)
            if obj_id not in self._dyn_objects and (xml is None or not xml.in_use):
                self._dyn_objects[obj_id] = self._add_dynamic_body(o)
                spec_changed = True
        if spec_changed:
            # Recompiling preserves qpos/qvel/ctrl/mocap state; only addresses need re-indexing.
            self._model, self._data = self._spec.recompile(self._model, self._data)
            self._index_model()
        return spec_changed or parked

    def _add_dynamic_body(self, o: dict[str, Any]) -> _PhysObject:
        """Add a primitive body for a sandbox object that has no counterpart in the MJCF."""
        w, depth, h = (max(10.0, float(o[k])) / 1000.0 for k in ("width_mm", "depth_mm", "height_mm"))
        self._dyn_serial += 1
        name = f"dyn_{self._dyn_serial}"
        if o["shape"] == "sphere":
            gtype, size, rest_z = mujoco.mjtGeom.mjGEOM_SPHERE, [w / 2, 0, 0], w / 2
            volume = math.pi * w**3 / 6
        elif o["shape"] == "cylinder":
            gtype, size, rest_z = mujoco.mjtGeom.mjGEOM_CYLINDER, [w / 2, h / 2, 0], h / 2
            volume = math.pi * w * w * h / 4
        else:
            gtype, size, rest_z = mujoco.mjtGeom.mjGEOM_BOX, [w / 2, depth / 2, h / 2], h / 2
            volume = w * depth * h
        body = self._spec.worldbody.add_body(
            name=name, pos=[float(o["x_mm"]) / 1000.0, float(o["y_mm"]) / 1000.0, rest_z]
        )
        movable = bool(o.get("movable", True))
        if movable:
            body.add_freejoint()
        else:
            body.mocap = True
        body.add_geom(
            type=gtype,
            size=size,
            mass=max(0.03, min(1.5, 250.0 * volume)),  # light hollow toys
            contype=2,
            conaffinity=2,
            friction=[0.5, 0.005, 0.0001],
        )
        return _PhysObject(name, movable, rest_z, dynamic=True, signature=_signature(o), in_use=True)
