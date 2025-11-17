# Next Steps After `apt update`

You just ran `apt update`. Now continue with these exact commands:

## Continue ROS2 Installation

```bash
# Install ROS2
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
```

## Then Setup Workspace

```bash
# Create workspace (you can be in any directory for this)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Copy package from Windows
cp -r /mnt/c/Users/maksa/Projects/Robot/robot_gazebo src/

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly

# Build
colcon build --packages-select robot_gazebo
source install/setup.bash
```

## Verify

```bash
ros2 pkg list | grep robot_gazebo
```

If you see `robot_gazebo` in the list, you're good!



