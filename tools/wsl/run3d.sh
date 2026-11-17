#!/usr/bin/env bash
# Run the robot in the 3D simulation (robot/ros/run3d.py) with ROS, the YOLOE environment and the model; arguments
# pass through. Run tools/wsl/brain.sh first.
#   wsl -- bash tools/wsl/run3d.sh --interactive --gui                    # Gazebo window + http://localhost:8765
#   wsl -- bash tools/wsl/run3d.sh --interactive --gui --room "two rooms"
#   wsl -- bash tools/wsl/run3d.sh --interactive --gui --forget           # start with an empty memory
repo="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="${SPROCKET_MODELS_DIR:-$HOME/sprocket-models}"
pkill -9 -f "[r]obot.ros.run3d" || true  # an earlier run still holding the simulator
for pat in "[g]z-sim-main" "[g]z-sim-gui" "[p]arameter_bridge" "[i]mage_bridge" "[c]omponent_container" "[r]obot.ros.perceive"; do
  pkill -9 -f "$pat" || true
done
source /opt/ros/lyrical/setup.bash
. ~/sprocket-venv/bin/activate
export PYTHONPATH="$repo:${PYTHONPATH:-}"  # appended: ROS's own Python packages are on it
cd "$MODELS_DIR" || { echo "missing models dir: $MODELS_DIR (run tools/wsl/setup_models.sh)"; exit 1; }
python -u -m robot.ros.run3d --brain http://127.0.0.1:8081 "$@" 2>&1 \
  | grep --line-buffered -v "^\[INFO\]\|never retrieved\|FutureWarning\|warnings.warn\|TF_OLD_DATA\|Possible reasons"
