"""3-CSI Pure-Vision perception, multi-zone ToF depth/step hazards, human pose, and tracking."""

from robot.perception.depth import StepHazard, calibrate_depth, classify_tof_grid, synthetic_scan
from robot.perception.locate import Place, lidar_near, lidar_range, locate, near_face
from robot.perception.overlay import draw_set_of_mark
from robot.perception.pose import HumanPose, estimate_pose_from_keypoints, resolve_ray_target
from robot.perception.rig import RIG, Camera, to_map, to_robot
from robot.perception.tracker import Detection, Track, Tracker, canonical, look, needs_active_vision

__all__ = [
    "RIG",
    "Camera",
    "Detection",
    "HumanPose",
    "Place",
    "StepHazard",
    "Track",
    "Tracker",
    "calibrate_depth",
    "canonical",
    "classify_tof_grid",
    "draw_set_of_mark",
    "estimate_pose_from_keypoints",
    "lidar_near",
    "lidar_range",
    "locate",
    "look",
    "near_face",
    "needs_active_vision",
    "resolve_ray_target",
    "synthetic_scan",
    "to_map",
    "to_robot",
]
