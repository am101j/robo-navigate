# Robot Gazebo Package

A comprehensive ROS2 package for a virtual robot in Gazebo with camera, LIDAR sensors, object detection, obstacle avoidance, Kafka streaming, and real-time visualization.

## Features

- **Virtual Robot**: Box-shaped robot with camera and 2D LIDAR sensors in Gazebo
- **Object Detection**: PyTorch-based GPU-accelerated object detection from camera images
- **Obstacle Detection**: Multi-threaded C++ LIDAR processing for real-time obstacle detection
- **Obstacle Avoidance**: AI-powered navigation with potential field algorithm
- **Kafka Streaming**: Real-time sensor data streaming to Kafka topics
- **Interactive Visualization**: Plotly dashboards for live sensor data monitoring
- **GPU Acceleration**: CUDA support for PyTorch object detection
- **Multi-threading**: Parallel processing in C++ for high-performance LIDAR analysis

## Package Structure

```
robot_gazebo/
├── CMakeLists.txt
├── package.xml
├── README.md
├── launch/
│   └── spawn_robot.launch.py
├── src/
│   ├── robot_gazebo/
│   │   ├── __init__.py
│   │   ├── camera_publisher.py          # Camera image publisher
│   │   ├── lidar_publisher.py           # LIDAR scan publisher
│   │   ├── object_detector.py           # GPU-accelerated object detection
│   │   ├── kafka_streamer.py           # Kafka producer for sensor data
│   │   ├── kafka_visualizer.py          # Plotly visualization from Kafka
│   │   └── obstacle_avoidance.py        # AI navigation controller
│   └── lidar_obstacle_detector.cpp      # Multi-threaded C++ LIDAR processor
└── urdf/
    └── robot.urdf.xacro                 # Robot model with sensors
```

## Dependencies

### ROS2 Packages
- ROS2 (tested with Humble/Humble Hawksbill)
- `rclcpp`, `rclpy` - ROS2 C++ and Python clients
- `sensor_msgs`, `geometry_msgs`, `std_msgs` - ROS2 message types
- `gazebo_ros`, `gazebo_ros_pkgs` - Gazebo integration
- `robot_state_publisher`, `joint_state_publisher` - Robot state
- `xacro` - URDF processing
- `cv_bridge` - OpenCV bridge for images

### Python Packages
```bash
pip install numpy opencv-python pillow torch torchvision kafka-python plotly
```

Or via package manager:
```bash
sudo apt install python3-numpy python3-opencv python3-pillow
pip3 install torch torchvision kafka-python plotly
```

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (optional, for object detection acceleration)
- **Kafka**: Apache Kafka server (optional, for streaming)
- **Gazebo**: Gazebo Classic or Gazebo with ROS2 integration

## Installation

### 1. Build the Package

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws

# Copy/clone this package to src/
cp -r /path/to/robot_gazebo src/

# Install ROS2 dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build the package
colcon build --packages-select robot_gazebo
source install/setup.bash
```

### 2. Install Python Dependencies

```bash
# Install via pip
pip3 install torch torchvision kafka-python plotly

# Or install PyTorch with CUDA support (for GPU acceleration)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Setup Kafka (Optional)

```bash
# Download and start Kafka
wget https://downloads.apache.org/kafka/2.13-3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# In another terminal, start Kafka
bin/kafka-server-start.sh config/server.properties
```

## Quick Start

### Launch Complete System

```bash
# Terminal 1: Launch Gazebo and spawn robot
ros2 launch robot_gazebo spawn_robot.launch.py

# Terminal 2: Start object detection (GPU-accelerated)
ros2 run robot_gazebo object_detector.py

# Terminal 3: Start LIDAR obstacle detection (multi-threaded C++)
ros2 run robot_gazebo lidar_obstacle_detector

# Terminal 4: Start obstacle avoidance navigation
ros2 run robot_gazebo obstacle_avoidance.py --ros-args \
  -p goal_x:=10.0 -p goal_y:=5.0

# Terminal 5: Stream to Kafka (optional)
ros2 run robot_gazebo kafka_streamer.py

# Terminal 6: Visualize with Plotly (optional)
ros2 run robot_gazebo kafka_visualizer.py
```

## Detailed Usage

### 1. Launch Gazebo Environment

Launch the robot in Gazebo with all sensors:

```bash
ros2 launch robot_gazebo spawn_robot.launch.py
```

This will:
- Start Gazebo simulator
- Spawn the virtual robot at (0, 0, 0.1)
- Initialize camera and LIDAR sensors
- Start camera and LIDAR publisher nodes

**Verify sensors are working:**
```bash
# Check topics
ros2 topic list

# View camera images
ros2 run rqt_image_view rqt_image_view
# Select topic: /camera/image_raw

# View LIDAR scans in RViz
ros2 run rviz2 rviz2
# Add LaserScan display, topic: /lidar/scan
```

### 2. Run ROS2 Nodes

#### Camera Publisher
Publishes camera images from Gazebo:
```bash
ros2 run robot_gazebo camera_publisher.py
```

#### LIDAR Publisher
Publishes LIDAR scan data:
```bash
ros2 run robot_gazebo lidar_publisher.py
```

#### Object Detection (GPU-Accelerated)
Performs real-time object detection using PyTorch:

```bash
# With default settings (auto-detects GPU)
ros2 run robot_gazebo object_detector.py

# Check GPU usage
nvidia-smi

# View detections
ros2 topic echo /camera/detections
```

**GPU Acceleration:**
- Automatically uses CUDA if available
- Falls back to CPU if GPU not available
- Uses Faster R-CNN ResNet-50 pre-trained model
- Detects 80 COCO object classes

**Output:** JSON messages on `/camera/detections` with:
- Object class names and IDs
- Confidence scores
- Bounding box coordinates

#### LIDAR Obstacle Detector (Multi-threaded C++)
High-performance obstacle detection using parallel processing:

```bash
# Run with default settings (uses all CPU cores)
ros2 run robot_gazebo lidar_obstacle_detector

# Customize thread count
ros2 run robot_gazebo lidar_obstacle_detector --ros-args \
  -p num_processing_threads:=4 \
  -p obstacle_threshold:=0.5

# View detected obstacles
ros2 topic echo /lidar/obstacles
```

**Multi-threading Features:**
- Parallel scan processing using worker threads
- Thread-safe queue for scan distribution
- Configurable thread count (default: CPU core count)
- Real-time obstacle clustering

**Performance:**
- Processes scans at 10 Hz
- Low latency with parallel processing
- Efficient memory management with bounded queues

#### Obstacle Avoidance (AI Navigation)
AI-powered navigation with obstacle avoidance:

```bash
# Basic usage
ros2 run robot_gazebo obstacle_avoidance.py

# With custom goal and parameters
ros2 run robot_gazebo obstacle_avoidance.py --ros-args \
  -p goal_x:=10.0 \
  -p goal_y:=5.0 \
  -p max_linear_velocity:=0.5 \
  -p max_angular_velocity:=1.0 \
  -p safety_distance:=1.5 \
  -p update_rate:=20.0

# View velocity commands
ros2 topic echo /cmd_vel
```

**AI Algorithm:**
- Potential field method for navigation
- Attractive force toward goal
- Repulsive forces from obstacles
- Sensor fusion (LIDAR + Camera)
- Emergency avoidance when obstacles too close

**Parameters:**
- `goal_x`, `goal_y`: Target position
- `safety_distance`: Minimum distance to obstacles
- `max_linear_velocity`: Maximum forward speed
- `max_angular_velocity`: Maximum turning speed
- `camera_detection_weight`: Camera influence (0.0-1.0)
- `lidar_detection_weight`: LIDAR influence (0.0-1.0)

### 3. Stream Sensor Data to Kafka

Stream sensor data to Kafka for external processing:

```bash
# Start Kafka server first (see Installation section)

# Start Kafka streamer
ros2 run robot_gazebo kafka_streamer.py --ros-args \
  -p kafka_bootstrap_servers:=localhost:9092 \
  -p kafka_camera_topic:=camera_detections \
  -p kafka_lidar_topic:=lidar_obstacles

# Verify messages in Kafka
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic camera_detections --from-beginning
```

**Kafka Topics:**
- `camera_detections`: Camera object detection results (JSON)
- `lidar_obstacles`: LIDAR obstacle positions (JSON)

**Message Format:**
```json
{
  "timestamp": 1234567890.123,
  "ros_timestamp": {"sec": 1234567890, "nanosec": 123000000},
  "source": "camera",
  "detections": [{"class_name": "person", "confidence": 0.85, ...}]
}
```

**Error Handling:**
- Automatic reconnection on connection loss
- Retry logic with exponential backoff
- Statistics tracking (sent/failed messages)
- Graceful degradation if Kafka unavailable

### 4. Visualize with Plotly Dashboards

Interactive real-time visualization of sensor data:

```bash
# Visualize from Kafka
ros2 run robot_gazebo kafka_visualizer.py

# Or use dummy data for testing (no Kafka required)
ros2 run robot_gazebo kafka_visualizer.py --dummy-data

# Custom Kafka settings
ros2 run robot_gazebo kafka_visualizer.py \
  --kafka-servers localhost:9092 \
  --camera-topic camera_detections \
  --lidar-topic lidar_obstacles
```

**Visualization Features:**
- **Robot Path & Obstacles**: 2D map showing robot trajectory and obstacles
- **Camera Detections Timeline**: Detection counts over time by object class
- **Obstacle Distribution**: Polar plot of obstacle distances and angles
- **Statistics Table**: Real-time message counts and metrics

**Interactive Features:**
- Zoom, pan, hover for details
- Live updates (500ms refresh rate)
- Color-coded data visualization
- Configurable history length

### 5. GPU Acceleration

**Object Detection GPU Usage:**

The object detection node automatically detects and uses GPU:

```bash
# Check if GPU is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Run object detector (will use GPU if available)
ros2 run robot_gazebo object_detector.py

# Monitor GPU usage
watch -n 1 nvidia-smi
```

**Performance:**
- GPU: ~30 FPS object detection
- CPU: ~5-10 FPS object detection
- Automatic fallback to CPU if GPU unavailable

### 6. Multi-threaded C++ Processing

**LIDAR Obstacle Detector Threading:**

The C++ LIDAR processor uses multi-threading for parallel scan processing:

```bash
# Check CPU usage (should show multiple threads)
htop

# Run with custom thread count
ros2 run robot_gazebo lidar_obstacle_detector --ros-args \
  -p num_processing_threads:=8
```

**Threading Architecture:**
- Main thread: Subscribes to LIDAR scans
- Worker threads: Process scans in parallel
- Thread-safe queue: Distributes work to threads
- Condition variables: Efficient thread synchronization

**Performance Benefits:**
- Parallel processing of scan data
- Reduced latency for obstacle detection
- Scalable with CPU core count
- Efficient memory usage

## ROS2 Topics

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera images (640x480 RGB, 30 Hz) |
| `/lidar/scan` | `sensor_msgs/LaserScan` | LIDAR scans (360 samples, 10 Hz) |
| `/camera/detections` | `std_msgs/String` | Object detections as JSON |
| `/lidar/obstacles` | `geometry_msgs/PointStamped` | Detected obstacle positions |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands for navigation |

### Subscribed Topics

| Topic | Type | Node |
|-------|------|------|
| `/camera/image_raw` | `sensor_msgs/Image` | `object_detector` |
| `/lidar/scan` | `sensor_msgs/LaserScan` | `lidar_obstacle_detector` |
| `/lidar/obstacles` | `geometry_msgs/PointStamped` | `obstacle_avoidance` |
| `/camera/detections` | `std_msgs/String` | `obstacle_avoidance`, `kafka_streamer` |

## Example Workflows

### Complete Navigation Pipeline

```bash
# Terminal 1: Gazebo
ros2 launch robot_gazebo spawn_robot.launch.py

# Terminal 2: Object Detection
ros2 run robot_gazebo object_detector.py

# Terminal 3: LIDAR Processing
ros2 run robot_gazebo lidar_obstacle_detector

# Terminal 4: Navigation
ros2 run robot_gazebo obstacle_avoidance.py --ros-args \
  -p goal_x:=10.0 -p goal_y:=0.0

# Terminal 5: Monitor
ros2 topic echo /cmd_vel
```

### Data Streaming Pipeline

```bash
# Terminal 1: Start Kafka
# (see Kafka setup instructions)

# Terminal 2: Gazebo + Sensors
ros2 launch robot_gazebo spawn_robot.launch.py

# Terminal 3: Object Detection
ros2 run robot_gazebo object_detector.py

# Terminal 4: LIDAR Processing
ros2 run robot_gazebo lidar_obstacle_detector

# Terminal 5: Kafka Streamer
ros2 run robot_gazebo kafka_streamer.py

# Terminal 6: Visualization
ros2 run robot_gazebo kafka_visualizer.py
```

### Testing Without Hardware

```bash
# Use dummy data mode for visualization
ros2 run robot_gazebo kafka_visualizer.py --dummy-data

# Test object detection with dummy camera images
# (camera_publisher generates test patterns)
ros2 run robot_gazebo object_detector.py
```

## Troubleshooting

### Gazebo Issues

**Gazebo not starting:**
```bash
# Check Gazebo installation
gazebo --version

# Install Gazebo ROS packages
sudo apt install ros-<distro>-gazebo-ros-pkgs
```

**Robot not spawning:**
```bash
# Check URDF validity
check_urdf urdf/robot.urdf.xacro

# Verify xacro is installed
sudo apt install ros-<distro>-xacro
```

### GPU Issues

**GPU not detected:**
```bash
# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python3 -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Low GPU utilization:**
- Ensure batch processing is enabled
- Check image resolution (lower = faster)
- Verify CUDA device is selected

### Kafka Issues

**Connection refused:**
```bash
# Check Kafka is running
jps | grep Kafka

# Verify port 9092 is open
netstat -tuln | grep 9092

# Check Kafka logs
tail -f kafka/logs/server.log
```

**No messages received:**
- Verify topics exist: `kafka-topics.sh --list --bootstrap-server localhost:9092`
- Check consumer offset: `kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group <group> --describe`

### Multi-threading Issues

**High CPU usage:**
- Reduce thread count: `-p num_processing_threads:=2`
- Check for thread contention in logs

**Low performance:**
- Increase thread count (up to CPU core count)
- Check queue size and processing rate

### Visualization Issues

**Plotly not displaying:**
- Install plotly: `pip3 install plotly`
- Check browser compatibility
- Try different renderer: modify code to use `renderer="browser"`

**No data in visualization:**
- Verify Kafka topics have data
- Check dummy data mode: `--dummy-data`
- Verify topic names match

## Performance Tuning

### Object Detection
- **GPU**: Use CUDA-enabled PyTorch for 3-5x speedup
- **Resolution**: Lower image resolution for faster processing
- **Model**: Use lighter models (MobileNet) for edge devices

### LIDAR Processing
- **Threads**: Set to CPU core count for optimal performance
- **Queue size**: Adjust based on scan rate
- **Clustering**: Disable for lower latency if not needed

### Navigation
- **Update rate**: Increase for more responsive control (10-20 Hz)
- **Safety distance**: Adjust based on robot speed
- **Sensor weights**: Tune based on sensor reliability

## Contributing

This package demonstrates:
- ROS2 Python and C++ node development
- Gazebo robot simulation
- GPU-accelerated AI inference
- Multi-threaded processing
- Real-time data streaming
- Interactive visualization

## License

MIT License

## Support

For issues and questions:
1. Check ROS2 topic list: `ros2 topic list`
2. Check node status: `ros2 node list`
3. View node logs: Check terminal output
4. Verify dependencies: `rosdep check --from-paths src`
