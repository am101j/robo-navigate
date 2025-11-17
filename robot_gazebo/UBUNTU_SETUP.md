# Ubuntu Linux Setup Guide

## Step 1: Check if ROS2 is Installed

```bash
# Check ROS2 version
ros2 --help

# Check which ROS2 distribution
echo $ROS_DISTRO
```

If ROS2 is installed, you'll see output. If not, continue to installation.

## Step 2: Install ROS2 (if needed)

### For Ubuntu 22.04 (Jammy) - ROS2 Humble:

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS2
sudo apt update
sudo apt install ros-humble-desktop -y

# Install build tools
sudo apt install python3-colcon-common-extensions python3-rosdep python3-argcomplete -y

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### For Ubuntu 20.04 (Focal) - ROS2 Foxy:

```bash
# Similar steps but use 'foxy' instead of 'humble'
sudo apt install ros-foxy-desktop -y
```

## Step 3: Verify ROS2 Installation

```bash
# Source ROS2 (if not already in .bashrc)
source /opt/ros/humble/setup.bash

# Test ROS2
ros2 --help
ros2 pkg list | head -5
```

## Step 4: Set Up Your Workspace

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Copy your package
# If you're in WSL, copy from Windows:
cp -r /mnt/c/Users/maksa/Projects/Robot/robot_gazebo src/

# Or if already in Linux:
# cp -r /path/to/robot_gazebo src/
```

## Step 5: Install Dependencies

```bash
cd ~/ros2_ws

# Install ROS2 dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Install Python dependencies
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly
```

## Step 6: Build the Package

```bash
cd ~/ros2_ws

# Build
colcon build --packages-select robot_gazebo

# Source the workspace
source install/setup.bash
```

## Step 7: Test It!

```bash
# Test 1: Check package is found
ros2 pkg list | grep robot_gazebo

# Test 2: Check executables
ros2 pkg executables robot_gazebo

# Test 3: Launch Gazebo (if installed)
ros2 launch robot_gazebo spawn_robot.launch.py
```

## Quick Commands Reference

```bash
# Always source ROS2 first
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Check topics
ros2 topic list

# Check nodes
ros2 node list

# View a topic
ros2 topic echo /camera/image_raw

# Run a node
ros2 run robot_gazebo camera_publisher.py
```

## Troubleshooting

### Issue: "ros2: command not found"
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Add to .bashrc permanently
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Issue: "Package not found"
```bash
# Rebuild and source
cd ~/ros2_ws
colcon build --packages-select robot_gazebo
source install/setup.bash
```

### Issue: Missing dependencies
```bash
# Install missing ROS2 packages
sudo apt install ros-humble-<package-name>

# Install Python packages
pip3 install <package-name>
```

## Next Steps

Once everything is set up:

1. **Launch the robot:**
   ```bash
   ros2 launch robot_gazebo spawn_robot.launch.py
   ```

2. **Run object detection:**
   ```bash
   ros2 run robot_gazebo object_detector.py
   ```

3. **Run navigation:**
   ```bash
   ros2 run robot_gazebo obstacle_avoidance.py
   ```

See QUICK_START.md for more detailed usage!



