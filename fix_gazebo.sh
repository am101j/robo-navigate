#!/bin/bash
# Quick fix script for Gazebo spawning issue

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_gazebo
source install/setup.bash

echo "Testing minimal launch..."
ros2 launch robot_gazebo gazebo_only.launch.py