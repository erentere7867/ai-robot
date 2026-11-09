"""Calibrate one of the robot's cameras (fisheye model) so its images can be rectified to the pinhole model in
config/body.yaml. Print a chessboard (9 x 6 inner corners, 25 mm squares), hold it at different angles and
distances all over the image, and press space for each view (20 or more); q finishes.

    python scripts/ventuno/calibrate_camera.py rear_left            # CSI camera of that name (body.yaml)
    python scripts/ventuno/calibrate_camera.py front --source usb

Writes ~/robot-runtime/calibration/<camera>.npz (K, D, size), which robot/hardware/sensors.py loads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot.perception.geometry import default_rig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", help="camera name from config/body.yaml")
    parser.add_argument("--source", choices=["csi", "usb"], default="csi")
    parser.add_argument("--board", default="9x6", help="inner corners, columns x rows")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--size", default="1280x720", help="capture resolution")
    args = parser.parse_args()
    model = next((c for c in default_rig() if c.name == args.camera), None)
    if model is None:
        parser.error(f"no camera named {args.camera} in config/body.yaml")
    cols, rows = (int(v) for v in args.board.split("x"))
    width, height = (int(v) for v in args.size.split("x"))
    if args.source == "csi":
        cap = cv2.VideoCapture(f"qtiqmmfsrc camera={model.device} ! video/x-raw,format=NV12,width={width},"
                               f"height={height},framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! "
                               "appsink drop=true max-buffers=1", cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(model.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    board = np.zeros((1, cols * rows, 3), np.float64)
    board[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm
    objects, images = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            print("The camera is not delivering frames.")
            return 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows),
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, (cols, rows), corners, found)
        cv2.putText(view, f"{len(images)} views (space: take, q: finish)", (20, 40), 0, 1.0, (0, 255, 0), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            objects.append(board.copy())
            images.append(corners.reshape(1, -1, 2).astype(np.float64))
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    if len(images) < 10:
        print("Need at least 10 views.")
        return 1
    k, d = np.zeros((3, 3)), np.zeros((4, 1))
    rms, k, d, _, _ = cv2.fisheye.calibrate(objects, images, (width, height), k, d,
                                            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
    out = Path("~/robot-runtime/calibration").expanduser() / f"{model.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=k, D=d, size=np.array([width, height]))
    print(f"RMS reprojection error {rms:.3f} px over {len(images)} views; saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
