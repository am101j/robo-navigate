# Exact Commands to Run This Package

Copy and paste these commands **one by one** in your Ubuntu terminal.

## STEP 1: Install ROS2 (if not installed)

```bash
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
```

**Wait for installation to finish** (takes 5-10 minutes)

## STEP 2: Create workspace and copy package

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

**Copy your package:**
- If using WSL (Windows files at `/mnt/c/`):
  ```bash
  cp -r /mnt/c/Users/maksa/Projects/Robot/robot_gazebo src/
  ```

- If package is already in Linux:
  ```bash
  cp -r ~/robot_gazebo src/  # or wherever it is
  ```

## STEP 3: Install dependencies

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly
```

## STEP 4: Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select robot_gazebo
source install/setup.bash
```

## STEP 5: Verify it works

```bash
ros2 pkg list | grep robot_gazebo
ros2 pkg executables robot_gazebo
```

You should see your nodes listed.

## STEP 6: RUN IT!

**Terminal 1 - Launch robot:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch robot_gazebo spawn_robot.launch.py
```

**Terminal 2 - Object detection:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_gazebo object_detector.py
```

**Terminal 3 - LIDAR processing:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_gazebo lidar_obstacle_detector
```

**Terminal 4 - Navigation:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_gazebo obstacle_avoidance.py --ros-args -p goal_x:=10.0 -p goal_y:=0.0
```

**Terminal 5 - Check it's working:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 topic list
ros2 topic echo /cmd_vel
```

## Quick Test (without Gazebo)

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_gazebo kafka_visualizer.py --dummy-data
```

This shows fake data in plots - no Gazebo needed!

## If Something Breaks

**"ros2: command not found"**
```bash
source /opt/ros/humble/setup.bash
```

**"Package not found"**
```bash
cd ~/ros2_ws
colcon build --packages-select robot_gazebo
source install/setup.bash
```

**"Permission denied"**
```bash
chmod +x src/robot_gazebo/src/robot_gazebo/*.py
```

That's it. Copy-paste and run.



