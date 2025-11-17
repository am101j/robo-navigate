#!/bin/bash
# Exact commands to run this package in Ubuntu
# Copy and paste these commands one by one

# ============================================
# STEP 1: Install ROS2 (if not installed)
# ============================================
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository universe -y
sudo apt update && sudo apt install curl gnupg lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
sudo apt install ros-humble-desktop -y
sudo apt install python3-colcon-common-extensions python3-rosdep -y
sudo rosdep init
rosdep update
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# ============================================
# STEP 2: Create workspace and copy package
# ============================================
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
# Copy package (adjust path if needed)
cp -r /mnt/c/Users/maksa/Projects/Robot/robot_gazebo src/ 2>/dev/null || cp -r ~/robot_gazebo src/ 2>/dev/null || echo "Copy robot_gazebo folder to ~/ros2_ws/src/ manually"

# ============================================
# STEP 3: Install dependencies
# ============================================
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly

# ============================================
# STEP 4: Build the package
# ============================================
cd ~/ros2_ws
colcon build --packages-select robot_gazebo
source install/setup.bash

# ============================================
# STEP 5: Test it works
# ============================================
ros2 pkg list | grep robot_gazebo
ros2 pkg executables robot_gazebo

echo "Done! Now run: ros2 launch robot_gazebo spawn_robot.launch.py"



