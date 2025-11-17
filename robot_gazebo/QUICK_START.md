# Quick Start Guide

## What This Package Does

This package creates a **simulated robot** that:

1. **Sees the world** with a camera and LIDAR sensor
2. **Detects objects** using AI (person, car, etc.) from camera images
3. **Finds obstacles** using LIDAR scans
4. **Navigates safely** by avoiding obstacles while moving toward a goal
5. **Streams data** to Kafka (optional) for external processing
6. **Visualizes everything** in real-time plots

Think of it as a **virtual robot** that can see, think, and move around in a simulated world.

## System Architecture

```
Gazebo Simulator
    ↓
Virtual Robot (Camera + LIDAR)
    ↓
┌─────────────────┬─────────────────┐
│                 │                 │
Camera Images    LIDAR Scans       │
    ↓                 ↓             │
Object Detection  Obstacle Detection│
(PyTorch GPU)    (Multi-threaded)  │
    ↓                 ↓             │
    └────────┬────────┘             │
             ↓                      │
    Obstacle Avoidance AI           │
             ↓                      │
    Velocity Commands (/cmd_vel)    │
             ↓                      │
    Robot Moves                     │
```

## Step-by-Step: How to Run It

### Prerequisites Check

```bash
# Check ROS2 is installed
ros2 --help

# Check Python 3
python3 --version

# Check if you have a ROS2 workspace
echo $ROS_DISTRO
```

### 1. Build the Package

```bash
# Navigate to your ROS2 workspace (create if needed)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Copy this package here (if not already there)
# Assuming you're in the Robot directory:
cp -r /path/to/robot_gazebo .  # Or just move it here

# Install dependencies
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select robot_gazebo

# Source the workspace
source install/setup.bash
```

### 2. Install Python Dependencies

```bash
# Install required Python packages
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly

# Or if you want GPU support (optional):
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Test Individual Components

#### Test 1: Launch Gazebo and See the Robot

```bash
# Terminal 1: Launch Gazebo
ros2 launch robot_gazebo spawn_robot.launch.py
```

**What to expect:**
- Gazebo window opens
- A blue box robot appears
- Camera and LIDAR sensors are active

**Verify it works:**
```bash
# In another terminal, check topics
ros2 topic list
# You should see: /camera/image_raw, /lidar/scan
```

#### Test 2: View Camera Images

```bash
# Terminal 2: View camera
ros2 run rqt_image_view rqt_image_view
# In the GUI, select topic: /camera/image_raw
```

**What to expect:**
- Image viewer window
- Camera feed (checkerboard pattern if using dummy data)

#### Test 3: View LIDAR Scans

```bash
# Terminal 3: View LIDAR in RViz
ros2 run rviz2 rviz2
```

**In RViz:**
1. Click "Add" → "By topic" → Select "LaserScan" → `/lidar/scan`
2. Click "Add" → "By display type" → "RobotModel"
3. Set Fixed Frame to `base_link` or `lidar_link`
4. You should see LIDAR scan visualization

#### Test 4: Object Detection (GPU)

```bash
# Terminal 4: Run object detection
ros2 run robot_gazebo object_detector.py
```

**What to expect:**
- Node starts and loads PyTorch model
- If GPU available: "Using device: cuda"
- If no GPU: "Using device: cpu"

**Check detections:**
```bash
# Terminal 5: View detections
ros2 topic echo /camera/detections
```

**What you'll see:**
```json
{
  "data": "[{\"class_name\": \"person\", \"confidence\": 0.85, ...}]"
}
```

#### Test 5: LIDAR Obstacle Detection (C++)

```bash
# Terminal 6: Run C++ obstacle detector
ros2 run robot_gazebo lidar_obstacle_detector
```

**What to expect:**
- "LIDAR Obstacle Detector Node started"
- "Using X processing threads" (X = your CPU cores)

**Check obstacles:**
```bash
# Terminal 7: View obstacles
ros2 topic echo /lidar/obstacles
```

**What you'll see:**
```
header:
  frame_id: "lidar_link"
point:
  x: 1.5
  y: 2.3
  z: 0.0
```

#### Test 6: Obstacle Avoidance Navigation

```bash
# Terminal 8: Run navigation
ros2 run robot_gazebo obstacle_avoidance.py --ros-args \
  -p goal_x:=5.0 \
  -p goal_y:=0.0
```

**What to expect:**
- Node starts with goal position
- Publishes velocity commands to `/cmd_vel`

**Check commands:**
```bash
# Terminal 9: View velocity commands
ros2 topic echo /cmd_vel
```

**What you'll see:**
```
linear:
  x: 0.3
  y: 0.0
  z: 0.0
angular:
  z: 0.1
```

### 4. Full System Test

Run everything together:

```bash
# Terminal 1: Gazebo + Robot
ros2 launch robot_gazebo spawn_robot.launch.py

# Terminal 2: Object Detection
ros2 run robot_gazebo object_detector.py

# Terminal 3: LIDAR Processing
ros2 run robot_gazebo lidar_obstacle_detector

# Terminal 4: Navigation
ros2 run robot_gazebo obstacle_avoidance.py --ros-args \
  -p goal_x:=10.0 -p goal_y:=5.0

# Terminal 5: Monitor
ros2 topic echo /cmd_vel
```

## Testing Without Gazebo (Dummy Data)

If you don't have Gazebo set up, you can test visualization:

```bash
# Test visualization with dummy data
ros2 run robot_gazebo kafka_visualizer.py --dummy-data
```

This will:
- Generate fake robot path (circular motion)
- Generate fake obstacles
- Generate fake camera detections
- Show interactive Plotly plots

## Common Issues & Solutions

### Issue: "Package not found"
```bash
# Make sure you sourced the workspace
source ~/ros2_ws/install/setup.bash

# Rebuild if needed
cd ~/ros2_ws
colcon build --packages-select robot_gazebo
source install/setup.bash
```

### Issue: "No module named torch"
```bash
pip3 install torch torchvision
```

### Issue: Gazebo doesn't start
```bash
# Install Gazebo
sudo apt install gazebo11 libgazebo11-dev

# Install ROS2 Gazebo packages
sudo apt install ros-$ROS_DISTRO-gazebo-ros-pkgs
```

### Issue: No topics appearing
```bash
# Check if nodes are running
ros2 node list

# Check if topics exist
ros2 topic list

# Check topic info
ros2 topic info /camera/image_raw
```

## Quick Verification Checklist

Run these commands to verify everything works:

```bash
# 1. Check package builds
colcon build --packages-select robot_gazebo
# Should see: "Finished <<< robot_gazebo"

# 2. Check nodes are available
ros2 pkg executables robot_gazebo
# Should list all Python and C++ nodes

# 3. Check topics (after launching)
ros2 topic list
# Should see: /camera/image_raw, /lidar/scan, /cmd_vel, etc.

# 4. Check nodes are running
ros2 node list
# Should see: camera_publisher, lidar_publisher, etc.

# 5. Check message rates
ros2 topic hz /camera/image_raw
# Should show ~30 Hz for camera

ros2 topic hz /lidar/scan
# Should show ~10 Hz for LIDAR
```

## What Success Looks Like

✅ **Gazebo**: Robot visible in simulation  
✅ **Camera**: Images publishing at ~30 Hz  
✅ **LIDAR**: Scans publishing at ~10 Hz  
✅ **Object Detection**: Detections appearing on `/camera/detections`  
✅ **Obstacle Detection**: Obstacles appearing on `/lidar/obstacles`  
✅ **Navigation**: Velocity commands on `/cmd_vel`  
✅ **Visualization**: Plotly plots updating in real-time  

## Next Steps

Once everything is working:

1. **Add obstacles to Gazebo** - Spawn boxes/spheres to test avoidance
2. **Change goal position** - Test navigation to different locations
3. **Tune parameters** - Adjust safety distance, speeds, etc.
4. **Add Kafka** - Stream data for external processing
5. **Customize visualization** - Modify Plotly plots for your needs

## Getting Help

If something doesn't work:

1. Check ROS2 logs in terminal output
2. Verify all dependencies are installed
3. Check `ros2 topic list` to see what's available
4. Check `ros2 node list` to see what's running
5. Review the full README.md for detailed documentation



