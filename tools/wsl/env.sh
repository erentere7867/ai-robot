# Source this in WSL before running any of Sprocket's ROS code:  . tools/wsl/env.sh
# ROS 2, the robot's Python environment, the repo on the path, OpenGL on the GPU, and ROS over shared memory.
# Models, maps and run output are in the repo's models/, maps/ and runs/ (robot/paths.py).
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/lyrical/setup.bash
. ~/sprocket-venv/bin/activate
export PYTHONPATH="$repo:${PYTHONPATH:-}"  # appended: ROS's own Python packages are on it
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"  # OpenGL on the GPU in WSL; without it Mesa draws on the CPU
export FASTDDS_DEFAULT_PROFILES_FILE="$repo/tools/wsl/fastdds_shm.xml"
