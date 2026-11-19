#!/usr/bin/env bash
# Set up a bare Ubuntu 26.04 machine to run and test Sprocket the way it is developed, in one go: system packages,
# ROS 2 lyrical with Nav2, the robot's Python environment, Habitat and its apartment, llama.cpp, and every model.
# Run it from the repository folder; it is safe to run again (each step skips what is already there).
#   bash tools/linux/setup.sh
# Then check the machine:  . tools/linux/env.sh && python -m robot.eval.doctor --start --3d
# Downloads about 7 GB and builds llama.cpp: 15 minutes on a 4-core cloud VM and a fast connection. Needs sudo or root.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
APT="$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q"
. /etc/os-release
step() { printf '\n=== %s\n' "$*"; }
if [ ! -f /opt/ros/lyrical/setup.bash ] && [ "${UBUNTU_CODENAME:-}" != resolute ]; then
  echo "ROS 2 lyrical, which the robot runs on, is packaged for Ubuntu 26.04 only, and this is $PRETTY_NAME." >&2
  echo "Use an Ubuntu 26.04 machine, VM or container (docs/testing.md shows one with Docker)." >&2
  exit 1
fi

step "1/6 system packages"
$SUDO apt-get update -q
$APT git curl ca-certificates software-properties-common build-essential cmake python3-venv python3-dev \
  ffmpeg mesa-utils libgl1 libegl1 bzip2

step "2/6 ROS 2 lyrical with Nav2 and slam_toolbox"
if [ ! -f /opt/ros/lyrical/setup.bash ]; then
  $SUDO add-apt-repository -y universe
  if ! dpkg -s ros2-apt-source > /dev/null 2>&1; then  # the ROS apt repository, as the ROS 2 install guide sets it up
    # Its newest tag, read with git: the guide's GitHub API call answers 60 times an hour per address and not at all
    # behind some proxies (a cloud VM got nothing, and set -e then stopped the script without a word).
    v=$(git ls-remote --tags --refs --sort=-v:refname https://github.com/ros-infrastructure/ros-apt-source \
      | awk -F/ 'NR == 1 {print $3}')
    curl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$v/\
ros2-apt-source_$v.${UBUNTU_CODENAME:-$VERSION_CODENAME}_all.deb"
    $SUDO dpkg -i /tmp/ros2-apt-source.deb
    $SUDO apt-get update -q
  fi
  $APT ros-lyrical-ros-base ros-lyrical-nav2-bringup ros-lyrical-slam-toolbox ros-lyrical-robot-localization \
    ros-lyrical-robot-state-publisher
fi

step "3/6 the robot's Python environment (~/sprocket-venv, the system Python, as ROS uses)"
# With the system's packages: ROS installs its Python dependencies with apt, and rclpy does not import without them
# ("No module named 'em'" on lyrical). The venv's own packages still come first. On a venv made without, it adds them.
python3 -m venv --system-site-packages ~/sprocket-venv
cpu_torch=""
command -v nvidia-smi > /dev/null || cpu_torch="--extra-index-url https://download.pytorch.org/whl/cpu"  # no NVIDIA
~/sprocket-venv/bin/pip install -q --upgrade pip                                # GPU: torch's CPU build, far smaller
~/sprocket-venv/bin/pip install -q $cpu_torch -e "$repo[eval,vision,voice,dev]"

step "4/6 Habitat-Sim and the apartment"
bash "$repo/tools/linux/setup_habitat.sh"

step "5/6 llama.cpp, the language model's server (~/llama.cpp)"
if [ ! -x ~/llama.cpp/build/bin/llama-server ] && [ ! -x ~/llama.cpp/build-cuda/bin/llama-server ]; then
  [ -d ~/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
  if command -v nvcc > /dev/null; then build=build-cuda; cuda=ON; else build=build; cuda=OFF; fi
  cmake -S ~/llama.cpp -B ~/llama.cpp/$build -DGGML_CUDA=$cuda -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build ~/llama.cpp/$build --target llama-server -j"$(nproc)"
fi

step "6/6 models (about 2.6 GB)"
bash "$repo/tools/linux/setup_models.sh"

echo
echo "Done. Check the machine:  . tools/linux/env.sh && python -m robot.eval.doctor --start --3d"
