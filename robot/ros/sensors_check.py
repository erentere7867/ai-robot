"""Check Sprocket's simulated hardware: launch it in a room and count what each sensor publishes.

    python3 -m robot.ros.sensors_check            # headless
    python3 -m robot.ros.sensors_check --gui      # with the Gazebo window

Prints messages per second for the 3 MIPI-CSI RGB-D cameras, front/rear ToF topics, synthetic 360-degree /scan,
odometry and the IMU, and checks whether AMCL localises the robot on the map (the map -> base_link transform).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

TOPICS = {  # topic: message type
    "/rgbd_camera/image": "sensor_msgs/msg/Image",
    "/rgbd_camera/depth_image": "sensor_msgs/msg/Image",
    "/rear_left_camera/image": "sensor_msgs/msg/Image",
    "/rear_left_camera/depth_image": "sensor_msgs/msg/Image",
    "/rear_right_camera/image": "sensor_msgs/msg/Image",
    "/rear_right_camera/depth_image": "sensor_msgs/msg/Image",
    "/tof/front": "std_msgs/msg/Float32MultiArray",
    "/tof/rear": "std_msgs/msg/Float32MultiArray",
    "/scan": "sensor_msgs/msg/LaserScan",
    "/odom": "nav_msgs/msg/Odometry",
    "/imu": "sensor_msgs/msg/Imu",
}


def main() -> None:
    import importlib

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    from robot.eval.suite import living_room
    from robot.ros.perceive import SyntheticScanPublisher
    from robot.ros.stage import launch, set_initial_pose
    from robot.sim.world import SimRobot

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--wait", type=float, default=40.0)
    args = parser.parse_args()
    world = SimRobot(living_room())
    sim = launch(world, Path(tempfile.mkdtemp(prefix="sprocket-check-")), gui=args.gui)
    try:
        rclpy.init()
        node = rclpy.create_node("sensors_check")
        SyntheticScanPublisher(node)
        counts = dict.fromkeys(TOPICS, 0)
        sizes: dict[str, str] = {}
        for topic, kind in TOPICS.items():
            package, _, name = kind.split("/")
            cls = getattr(importlib.import_module(f"{package}.msg"), name)

            def seen(msg: object, topic: str = topic) -> None:
                counts[topic] += 1
                if hasattr(msg, "width"):
                    sizes[topic] = f"{msg.width}x{msg.height} {msg.encoding} frame {msg.header.frame_id}"
                elif hasattr(msg, "ranges"):
                    sizes[topic] = f"{len(msg.ranges)} ranges, frame {msg.header.frame_id}"
                elif topic.startswith("/tof/") and hasattr(msg, "data"):
                    sizes[topic] = f"{len(msg.data)} zones"
            node.create_subscription(cls, topic, seen, qos_profile_sensor_data)
        tf = Buffer()
        TransformListener(tf, node)
        print(f"waiting {args.wait:.0f} s for Gazebo and Nav2...", flush=True)
        end = time.monotonic() + args.wait
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        set_initial_pose(node, world)
        for topic in counts:
            counts[topic] = 0
        start = time.monotonic()
        while time.monotonic() - start < 5.0:
            rclpy.spin_once(node, timeout_sec=0.05)
        for topic, n in counts.items():
            print(f"{topic:<30} {n / 5.0:5.1f} msg/s  {sizes.get(topic, '')}")
        try:
            t = tf.lookup_transform("map", "base_link", Time()).transform.translation
            print(f"localised: base_link at ({t.x:.2f}, {t.y:.2f}) in the map")
        except Exception as exc:  # noqa: BLE001
            print(f"not localised yet: {exc}")
        if args.gui:
            input("Press Enter to close the simulation...")
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        sim.stop()


if __name__ == "__main__":
    main()
