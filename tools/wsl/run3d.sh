#!/usr/bin/env bash
# Run the robot in the 3D simulation (robot/ros/run3d.py) with ROS, the YOLOE environment and the model; arguments
# pass through. Run tools/wsl/brain.sh first.
#   wsl -- bash tools/wsl/run3d.sh --interactive --gui                    # Gazebo window + http://localhost:8765
#   wsl -- bash tools/wsl/run3d.sh --interactive --gui --room "two rooms"
#   wsl -- bash tools/wsl/run3d.sh --interactive --gui --forget           # start with an empty memory
repo="$(cd "$(dirname "$0")/../.." && pwd)"
pkill -f "[r]obot.ros.run3d"  # an earlier run still holding the simulator
source /opt/ros/lyrical/setup.bash
. ~/sprocket-venv/bin/activate
export PYTHONPATH="$repo:$PYTHONPATH"  # appended: ROS's own Python packages are on it
cd ~/sprocket-models  # the YOLOE weights live here
python -u -m robot.ros.run3d --brain http://127.0.0.1:8081 "$@" 2>&1 \
  | grep --line-buffered -v "^\[INFO\]\|never retrieved\|FutureWarning\|warnings.warn\|TF_OLD_DATA\|Possible reasons"
