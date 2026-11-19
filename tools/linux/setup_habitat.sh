#!/usr/bin/env bash
# Set up the 3D apartment, in WSL or on any Ubuntu (robot/sim/habitat_server.py, run by robot/sim/stack.py): GPU
# rendering in WSL, Habitat-Sim in its own Python, the free scenes.
#   bash tools/linux/setup_habitat.sh
# Habitat-Sim 0.3.3 is built only for Python 3.9, so it lives in its own micromamba env (~/micromamba/envs/habitat)
# and the robot talks to it over a local socket. Downloads: micromamba (~15 MB), Habitat-Sim (~250 MB),
# Habitat test scenes (~106 MB) and ReplicaCAD (~302 MB, CC BY 4.0).
set -euo pipefail

# 1. In WSL, OpenGL on the GPU through Direct3D 12: WSL ships the driver, but Mesa picks the CPU unless asked. Elsewhere
#    the machine's own OpenGL driver is used (on a machine without a GPU, Mesa's software renderer: slow but it works).
if [ -e /dev/dxg ]; then
  line='export GALLIUM_DRIVER=d3d12  # render OpenGL on the GPU through Direct3D 12 (WSL); Mesa picks the CPU otherwise'
  for f in ~/.profile ~/.bashrc; do grep -q "GALLIUM_DRIVER=d3d12" "$f" || printf '\n%s\n' "$line" >> "$f"; done
  export GALLIUM_DRIVER=d3d12
fi
glxinfo -B 2>/dev/null | grep -E "renderer string" || echo "(glxinfo not installed: sudo apt install mesa-utils to check)"

# 2. micromamba and the habitat env (Python 3.9, Habitat-Sim with Bullet for furniture with doors, git-lfs).
export MAMBA_ROOT_PREFIX=~/micromamba
mkdir -p ~/.local/bin
[ -x ~/.local/bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/.local bin/micromamba
MM=~/.local/bin/micromamba
[ -d ~/micromamba/envs/habitat ] || $MM create -y -n habitat -c aihabitat -c conda-forge python=3.9 "habitat-sim=0.3.3" withbullet git-lfs
$MM run -n habitat git lfs install > /dev/null

# 3. The scenes.
if [ ! -d ~/habitat-data/replica_cad ]; then
  mkdir -p ~/habitat-data && cd ~/habitat-data
  $MM run -n habitat python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes replica_cad_dataset \
    --data-path ~/habitat-data
fi
$MM run -n habitat python -c "import habitat_sim; print('habitat-sim', habitat_sim.__version__, 'ready')"
