#!/usr/bin/env bash
# Stop the simulated home and the robot's software and, with --all, the model too.
#   wsl -- bash tools/linux/stop.sh --all
for p in robot.eval.home3d robot.interfaces.demo robot.sim.habitat_ros habitat_server.py robot.ros.perceive \
         component_container robot_state_publisher ekf_node slam_toolbox "ros2 launch"; do
  pkill -9 -f "[${p:0:1}]${p:1}"
done
if [ "$1" = "--all" ]; then
  pkill -f "[l]lama-server"
  for _ in 1 2 3; do pgrep -f "[l]lama-server" > /dev/null || break; sleep 1; done
  pkill -9 -f "[l]lama-server"
fi
echo "stopped"
