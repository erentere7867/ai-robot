#!/usr/bin/env bash
# Stop the simulation and, with --all, the model too.
#   wsl -- bash tools/wsl/stop.sh --all
pkill -9 -f "[r]obot.ros.run3d"
pkill -9 -f "[t]ools/wsl/run3d.sh"
for p in gz-sim-main gz-sim-gui "gz sim" component_container robot_state_publisher parameter_bridge image_bridge \
         "ros2 launch" rviz2 ekf_node; do
  pkill -9 -f "[${p:0:1}]${p:1}"
done
if [ "$1" = "--all" ]; then
  pkill -f "[l]lama-server"
  for _ in 1 2 3; do pgrep -f "[l]lama-server" > /dev/null || break; sleep 1; done
  pkill -9 -f "[l]lama-server"
fi
echo "stopped"
