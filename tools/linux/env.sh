# Source this (in WSL or on any Ubuntu) before running any of Sprocket's ROS code:  . tools/linux/env.sh
# ROS 2, the robot's Python environment, the repo on the path, OpenGL on the GPU in WSL, and ROS over shared memory.
# Models, maps and run output are in the repo's models/, maps/ and runs/ (robot/paths.py).
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/lyrical/setup.bash
. ~/sprocket-venv/bin/activate
export PYTHONPATH="$repo:${PYTHONPATH:-}"  # appended: ROS's own Python packages are on it
if [ -e /dev/dxg ]; then  # WSL: OpenGL on the GPU through Direct3D 12; without it Mesa draws on the CPU
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
fi
export FASTDDS_DEFAULT_PROFILES_FILE="$repo/tools/linux/fastdds_shm.xml"
