# Autonomous Robot Simulation

ROS2-based autonomous robot simulation with AI object detection, LIDAR obstacle avoidance, and real-time navigation in Gazebo.

## Features

- **3D Simulation**: Gazebo environment with room, walls, and obstacles
- **AI Object Detection**: PyTorch-based real-time object detection (GPU accelerated)
- **LIDAR Processing**: 360° laser scanning with multi-threaded obstacle detection
- **Autonomous Navigation**: Potential field-based obstacle avoidance
- **Real-time Visualization**: Camera feeds and sensor data visualization
- **Data Streaming**: Optional Kafka integration for external processing

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

## Quick Start

### Prerequisites

- Ubuntu 20.04/22.04 or WSL2
- ROS2 Humble
- Python 3.8+
- Gazebo 11

### Installation

```bash
# 1. Create ROS2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/yourusername/autonomous-robot-simulation.git robot_gazebo

# 2. Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
pip3 install numpy opencv-python pillow torch torchvision kafka-python plotly

# 3. Build package
colcon build --packages-select robot_gazebo
source install/setup.bash
```

### Running the Simulation

**Terminal 1: Launch Gazebo with world**
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 launch gazebo_ros gazebo.launch.py world:=~/ros2_ws/src/robot_gazebo/worlds/test_world.world
```

**Terminal 2: Spawn robot**
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run gazebo_ros spawn_entity.py -entity turtlebot3 -database turtlebot3_burger -x 0 -y 0 -z 0.1
```

**Terminal 3: Camera system**
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run robot_gazebo camera_publisher.py
```

**Terminal 4: LIDAR system**
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run robot_gazebo lidar_publisher.py
```

**Terminal 5: LIDAR processing**
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run robot_gazebo lidar_obstacle_detector
```

**Terminal 6: Autonomous navigation**
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run robot_gazebo obstacle_avoidance.py --ros-args -p goal_x:=4.0 -p goal_y:=3.0
```

### Manual Robot Control

```bash
# Move forward
ros2 topic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.2}" --once

# Drive in circle
ros2 topic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.1}, angular: {z: 0.3}" -r 10
```

## Project Structure

```
robot_gazebo/
├── src/robot_gazebo/
│   ├── camera_publisher.py      # Camera image processing
│   ├── lidar_publisher.py       # LIDAR data generation
│   ├── object_detector.py       # AI object detection
│   ├── obstacle_avoidance.py    # Navigation algorithm
│   ├── robot_controller.py      # Manual control
│   ├── kafka_streamer.py        # Data streaming
│   └── kafka_visualizer.py      # Real-time plots
├── launch/
│   ├── gazebo_robot.launch.py   # Complete system launch
│   └── spawn_robot.launch.py    # Robot spawning
├── worlds/
│   └── test_world.world         # Gazebo simulation world
├── urdf/
│   └── robot.urdf.xacro         # Robot model
└── CMakeLists.txt
```

## Key Components

### Object Detection
- **Model**: Faster R-CNN ResNet50 FPN
- **Classes**: 80 COCO classes (person, car, etc.)
- **Performance**: GPU accelerated inference
- **Output**: JSON detection messages

### LIDAR Processing
- **Range**: 360° scanning, 0.1-10m range
- **Processing**: Multi-threaded C++ implementation
- **Features**: Real-time obstacle detection
- **Output**: Obstacle coordinates

### Navigation Algorithm
- **Method**: Potential field-based navigation
- **Features**: Goal attraction + obstacle repulsion
- **Safety**: Configurable safety distances
- **Performance**: 10Hz control loop

## Configuration

Key parameters in `obstacle_avoidance.py`:
```python
max_linear_velocity: 0.5    # m/s
max_angular_velocity: 1.0   # rad/s
safety_distance: 1.0        # m
goal_tolerance: 0.5         # m
```

## Troubleshooting

### Common Issues

**"Package not found"**
```bash
cd ~/ros2_ws
colcon build --packages-select robot_gazebo
source install/setup.bash
```

**"No module named torch"**
```bash
pip3 install torch torchvision
```

**Robot doesn't move**
```bash
# Check if robot is spawned
ros2 topic list | grep cmd_vel
# Test manual movement
ros2 topic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.2}" --once
```

### Verification Commands

```bash
# Check all nodes running
ros2 node list

# Check topic rates
ros2 topic hz /lidar/scan
ros2 topic hz /camera/image_raw

# Monitor robot commands
ros2 topic echo /cmd_vel
```

## Demo Without Gazebo

Test with simulated data:
```bash
ros2 run robot_gazebo kafka_visualizer.py --dummy-data
```

## Hardware Requirements

- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU, GPU for object detection
- **GPU**: NVIDIA GPU with CUDA for PyTorch acceleration

## Dependencies

- ROS2 Humble
- Gazebo 11
- Python packages: numpy, opencv-python, torch, torchvision, kafka-python, plotly
- C++ libraries: Standard ROS2 packages

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Acknowledgments

- ROS2 community for robotics framework
- PyTorch team for deep learning tools
- Gazebo simulator developers