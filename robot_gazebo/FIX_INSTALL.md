# Fix ROS2 Installation Error

The error means ROS2 Humble didn't install. Let's check your Ubuntu version and install the right ROS2.

## Check Your Ubuntu Version

```bash
lsb_release -a
```

## If Ubuntu 24.04 → Install ROS2 Jazzy

Ubuntu 24.04 uses ROS2 Jazzy, not Humble. Run this:

```bash
sudo apt update
sudo apt install ros-jazzy-desktop -y
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## If Ubuntu 22.04 → Install ROS2 Humble

```bash
sudo apt update
sudo apt install ros-humble-desktop -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Verify Installation

```bash
# Check ROS2 is installed
ls /opt/ros/

# You should see: humble, jazzy, or both

# Test ROS2
source /opt/ros/humble/setup.bash  # or jazzy
ros2 --help
```

## Then Continue Setup

After ROS2 is installed, continue with workspace setup:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
cp -r /mnt/c/Users/maksa/Projects/Robot/robot_gazebo src/
rosdep install --from-paths src --ignore-src -r -y
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly
colcon build --packages-select robot_gazebo
source install/setup.bash
```



