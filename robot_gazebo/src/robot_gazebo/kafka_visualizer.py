#!/usr/bin/env python3
"""
Kafka Visualizer
Reads sensor data from Kafka topics and visualizes:
- Robot path
- Detected obstacles (LIDAR)
- Detected objects (Camera)
Uses Plotly for interactive visualization with live updates.
"""

import json
import time
import threading
import argparse
from collections import deque
from typing import List, Dict, Optional, Tuple
import math
import random

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Will use dummy data mode.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Error: plotly not installed. Install with: pip install plotly")
    exit(1)


class SensorDataVisualizer:
    """Visualizes sensor data from Kafka topics"""
    
    def __init__(self, kafka_servers: str = 'localhost:9092', 
                 camera_topic: str = 'camera_detections',
                 lidar_topic: str = 'lidar_obstacles',
                 use_dummy_data: bool = False,
                 max_history: int = 1000):
        self.kafka_servers = kafka_servers
        self.camera_topic = camera_topic
        self.lidar_topic = lidar_topic
        self.use_dummy_data = use_dummy_data or not KAFKA_AVAILABLE
        self.max_history = max_history
        
        # Data storage
        self.robot_path: deque = deque(maxlen=max_history)
        self.obstacles: deque = deque(maxlen=max_history)
        self.camera_objects: deque = deque(maxlen=max_history)
        
        # Threading
        self.running = False
        self.consumer_thread: Optional[threading.Thread] = None
        self.dummy_data_thread: Optional[threading.Thread] = None
        
        # Robot position tracking (simulated from obstacles)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.path_timestamp = time.time()
        
        # Statistics
        self.camera_messages_received = 0
        self.lidar_messages_received = 0
        
        print(f"Initialized visualizer - Dummy data mode: {self.use_dummy_data}")
    
    def start(self):
        """Start consuming data and visualization"""
        self.running = True
        
        if self.use_dummy_data:
            print("Starting dummy data simulation...")
            self.dummy_data_thread = threading.Thread(target=self._generate_dummy_data, daemon=True)
            self.dummy_data_thread.start()
        else:
            print(f"Connecting to Kafka at {self.kafka_servers}...")
            try:
                self.consumer = KafkaConsumer(
                    self.camera_topic,
                    self.lidar_topic,
                    bootstrap_servers=[self.kafka_servers],
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    consumer_timeout_ms=1000
                )
                print("Connected to Kafka successfully")
                self.consumer_thread = threading.Thread(target=self._consume_kafka, daemon=True)
                self.consumer_thread.start()
            except Exception as e:
                print(f"Failed to connect to Kafka: {e}")
                print("Falling back to dummy data mode...")
                self.use_dummy_data = True
                self.dummy_data_thread = threading.Thread(target=self._generate_dummy_data, daemon=True)
                self.dummy_data_thread.start()
        
        # Start visualization
        self._visualize()
    
    def _consume_kafka(self):
        """Consume messages from Kafka topics"""
        while self.running:
            try:
                message_pack = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    topic = topic_partition.topic
                    
                    for message in messages:
                        try:
                            data = message.value
                            
                            if topic == self.camera_topic:
                                self._process_camera_data(data)
                            elif topic == self.lidar_topic:
                                self._process_lidar_data(data)
                                
                        except Exception as e:
                            print(f"Error processing message: {e}")
                            
            except KafkaError as e:
                print(f"Kafka error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Unexpected error in Kafka consumer: {e}")
                time.sleep(1)
    
    def _process_camera_data(self, data: Dict):
        """Process camera detection data"""
        self.camera_messages_received += 1
        
        if 'detections' in data:
            detections = data['detections']
            if isinstance(detections, list):
                for detection in detections:
                    camera_obj = {
                        'timestamp': data.get('timestamp', time.time()),
                        'class_name': detection.get('class_name', 'unknown'),
                        'class_id': detection.get('class_id', 0),
                        'confidence': detection.get('confidence', 0.0),
                        'bbox': detection.get('bbox', {}),
                        'robot_x': self.robot_x,
                        'robot_y': self.robot_y
                    }
                    self.camera_objects.append(camera_obj)
    
    def _process_lidar_data(self, data: Dict):
        """Process LIDAR obstacle data"""
        self.lidar_messages_received += 1
        
        if 'obstacle' in data:
            obstacle = data['obstacle']
            # Convert obstacle from LIDAR frame to world frame (simplified)
            lidar_x = obstacle.get('x', 0.0)
            lidar_y = obstacle.get('y', 0.0)
            
            # Transform to world coordinates (assuming robot at origin initially)
            world_x = self.robot_x + lidar_x * math.cos(self.robot_theta) - lidar_y * math.sin(self.robot_theta)
            world_y = self.robot_y + lidar_x * math.sin(self.robot_theta) + lidar_y * math.cos(self.robot_theta)
            
            obstacle_data = {
                'timestamp': data.get('timestamp', time.time()),
                'x': world_x,
                'y': world_y,
                'z': obstacle.get('z', 0.0),
                'robot_x': self.robot_x,
                'robot_y': self.robot_y
            }
            self.obstacles.append(obstacle_data)
            
            # Update robot path (simulate movement based on obstacles)
            self._update_robot_path()
    
    def _update_robot_path(self):
        """Update robot path based on simulated movement"""
        current_time = time.time()
        dt = current_time - self.path_timestamp
        self.path_timestamp = current_time
        
        # Simple simulated movement (circular path)
        if dt > 0:
            # Move in a circle
            radius = 5.0
            angular_velocity = 0.1  # rad/s
            self.robot_theta += angular_velocity * dt
            self.robot_x = radius * math.cos(self.robot_theta)
            self.robot_y = radius * math.sin(self.robot_theta)
            
            # Add to path
            path_point = {
                'timestamp': current_time,
                'x': self.robot_x,
                'y': self.robot_y,
                'theta': self.robot_theta
            }
            self.robot_path.append(path_point)
    
    def _generate_dummy_data(self):
        """Generate dummy sensor data for testing"""
        print("Generating dummy sensor data...")
        
        start_time = time.time()
        obstacle_id = 0
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update robot path (circular motion)
            radius = 5.0
            angular_velocity = 0.1
            self.robot_theta = angular_velocity * elapsed
            self.robot_x = radius * math.cos(self.robot_theta)
            self.robot_y = radius * math.sin(self.robot_theta)
            
            # Add path point
            path_point = {
                'timestamp': current_time,
                'x': self.robot_x,
                'y': self.robot_y,
                'theta': self.robot_theta
            }
            self.robot_path.append(path_point)
            
            # Generate obstacles (random around robot)
            if random.random() < 0.3:  # 30% chance per iteration
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(1.0, 4.0)
                obstacle_x = self.robot_x + distance * math.cos(angle)
                obstacle_y = self.robot_y + distance * math.sin(angle)
                
                obstacle_data = {
                    'timestamp': current_time,
                    'x': obstacle_x,
                    'y': obstacle_y,
                    'z': 0.0,
                    'robot_x': self.robot_x,
                    'robot_y': self.robot_y
                }
                self.obstacles.append(obstacle_data)
                obstacle_id += 1
            
            # Generate camera detections (periodic)
            if int(elapsed) % 2 == 0 and random.random() < 0.5:
                classes = ['person', 'car', 'bicycle', 'dog', 'chair']
                class_name = random.choice(classes)
                
                detection = {
                    'timestamp': current_time,
                    'class_name': class_name,
                    'class_id': classes.index(class_name) + 1,
                    'confidence': random.uniform(0.6, 0.95),
                    'bbox': {
                        'x_min': random.uniform(0.1, 0.4),
                        'y_min': random.uniform(0.1, 0.4),
                        'x_max': random.uniform(0.5, 0.9),
                        'y_max': random.uniform(0.5, 0.9)
                    },
                    'robot_x': self.robot_x,
                    'robot_y': self.robot_y
                }
                self.camera_objects.append(detection)
                self.camera_messages_received += 1
            
            time.sleep(0.1)  # 10 Hz update rate
    
    def _visualize(self):
        """Create and update interactive Plotly visualization"""
        print("Starting visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Robot Path & Obstacles', 'Camera Detections Timeline', 
                          'Obstacle Distribution', 'Statistics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        update_interval = 0.5  # Update every 500ms
        
        while self.running:
            try:
                # Clear previous traces (except table)
                fig.data = []
                
                # 1. Robot Path & Obstacles (Top Left)
                if len(self.robot_path) > 0:
                    path_x = [p['x'] for p in self.robot_path]
                    path_y = [p['y'] for p in self.robot_path]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=path_x, y=path_y,
                            mode='lines+markers',
                            name='Robot Path',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        ),
                        row=1, col=1
                    )
                    
                    # Current robot position
                    if len(self.robot_path) > 0:
                        latest = self.robot_path[-1]
                        fig.add_trace(
                            go.Scatter(
                                x=[latest['x']], y=[latest['y']],
                                mode='markers',
                                name='Robot Position',
                                marker=dict(size=15, color='red', symbol='circle')
                            ),
                            row=1, col=1
                        )
                
                # Obstacles
                if len(self.obstacles) > 0:
                    obs_x = [o['x'] for o in self.obstacles]
                    obs_y = [o['y'] for o in self.obstacles]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=obs_x, y=obs_y,
                            mode='markers',
                            name='Obstacles',
                            marker=dict(size=8, color='orange', symbol='x')
                        ),
                        row=1, col=1
                    )
                
                # 2. Camera Detections Timeline (Top Right)
                if len(self.camera_objects) > 0:
                    # Group by class
                    class_counts = {}
                    timestamps = []
                    for obj in list(self.camera_objects)[-50:]:  # Last 50 detections
                        class_name = obj['class_name']
                        if class_name not in class_counts:
                            class_counts[class_name] = []
                        class_counts[class_name].append(obj['timestamp'])
                        timestamps.append(obj['timestamp'])
                    
                    # Plot detection counts over time
                    if timestamps:
                        time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
                        bins = 20
                        bin_size = time_range / bins if time_range > 0 else 1
                        
                        for class_name, times in class_counts.items():
                            counts, edges = self._histogram(times, bins, min(timestamps), max(timestamps))
                            fig.add_trace(
                                go.Scatter(
                                    x=edges[:-1], y=counts,
                                    mode='lines+markers',
                                    name=f'{class_name}',
                                    stackgroup='one'
                                ),
                                row=1, col=2
                            )
                
                # 3. Obstacle Distribution (Bottom Left)
                if len(self.obstacles) > 0:
                    # Distance from robot
                    distances = []
                    angles = []
                    for obs in self.obstacles:
                        dx = obs['x'] - obs['robot_x']
                        dy = obs['y'] - obs['robot_y']
                        dist = math.sqrt(dx*dx + dy*dy)
                        angle = math.atan2(dy, dx)
                        distances.append(dist)
                        angles.append(angle)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=angles, y=distances,
                            mode='markers',
                            name='Obstacle Distance',
                            marker=dict(size=6, color='purple', opacity=0.6)
                        ),
                        row=2, col=1
                    )
                
                # 4. Statistics Table (Bottom Right)
                stats_data = [
                    ['Camera Messages', str(self.camera_messages_received)],
                    ['LIDAR Messages', str(self.lidar_messages_received)],
                    ['Path Points', str(len(self.robot_path))],
                    ['Obstacles', str(len(self.obstacles))],
                    ['Camera Objects', str(len(self.camera_objects))],
                    ['Mode', 'Dummy Data' if self.use_dummy_data else 'Kafka']
                ]
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric', 'Value'],
                                  fill_color='paleturquoise',
                                  align='left'),
                        cells=dict(values=[[row[0] for row in stats_data],
                                        [row[1] for row in stats_data]],
                                 fill_color='lavender',
                                 align='left')
                    ),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_xaxes(title_text="X (m)", row=1, col=1)
                fig.update_yaxes(title_text="Y (m)", row=1, col=1)
                fig.update_xaxes(title_text="Time", row=1, col=2)
                fig.update_yaxes(title_text="Detection Count", row=1, col=2)
                fig.update_xaxes(title_text="Angle (rad)", row=2, col=1)
                fig.update_yaxes(title_text="Distance (m)", row=2, col=1)
                
                fig.update_layout(
                    title_text="Robot Sensor Data Visualization",
                    height=900,
                    showlegend=True
                )
                
                # Show/update plot
                fig.show(renderer="browser" if len(self.robot_path) == 1 else None)
                
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                print("\nStopping visualization...")
                self.running = False
                break
            except Exception as e:
                print(f"Error in visualization: {e}")
                time.sleep(update_interval)
    
    def _histogram(self, data: List[float], bins: int, min_val: float, max_val: float) -> Tuple[List[int], List[float]]:
        """Simple histogram calculation"""
        if not data or max_val == min_val:
            return [0] * bins, [min_val] * (bins + 1)
        
        bin_size = (max_val - min_val) / bins
        counts = [0] * bins
        edges = [min_val + i * bin_size for i in range(bins + 1)]
        
        for value in data:
            if min_val <= value <= max_val:
                bin_idx = min(int((value - min_val) / bin_size), bins - 1)
                counts[bin_idx] += 1
        
        return counts, edges
    
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        if hasattr(self, 'consumer'):
            try:
                self.consumer.close()
            except:
                pass
        print("Visualizer stopped")


def main():
    parser = argparse.ArgumentParser(description='Visualize robot sensor data from Kafka')
    parser.add_argument('--kafka-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--camera-topic', type=str, default='camera_detections',
                        help='Kafka topic for camera detections')
    parser.add_argument('--lidar-topic', type=str, default='lidar_obstacles',
                        help='Kafka topic for LIDAR obstacles')
    parser.add_argument('--dummy-data', action='store_true',
                        help='Use dummy data instead of Kafka')
    parser.add_argument('--max-history', type=int, default=1000,
                        help='Maximum number of data points to keep in history')
    
    args = parser.parse_args()
    
    visualizer = SensorDataVisualizer(
        kafka_servers=args.kafka_servers,
        camera_topic=args.camera_topic,
        lidar_topic=args.lidar_topic,
        use_dummy_data=args.dummy_data,
        max_history=args.max_history
    )
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        visualizer.stop()


if __name__ == '__main__':
    main()



