#!/usr/bin/env python3
"""
Obstacle Avoidance Node
Subscribes to /lidar/obstacles and /camera/detections to implement
obstacle avoidance navigation. Publishes velocity commands to /cmd_vel.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String
import json
import math
import time
from collections import deque
from typing import List, Dict, Optional


class ObstacleAvoidance(Node):
    """ROS2 node for obstacle avoidance navigation"""
    
    def __init__(self):
        super().__init__('obstacle_avoidance')
        
        # Declare parameters
        self.declare_parameter('max_linear_velocity', 0.5)  # m/s
        self.declare_parameter('max_angular_velocity', 1.0)  # rad/s
        self.declare_parameter('safety_distance', 0.2)  # m
        self.declare_parameter('obstacle_influence_radius', 0.8)  # m
        self.declare_parameter('camera_detection_weight', 0.3)
        self.declare_parameter('lidar_detection_weight', 0.7)
        self.declare_parameter('goal_x', 10.0)  # Goal position
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.5)  # m
        self.declare_parameter('update_rate', 10.0)  # Hz
        
        # Get parameters
        self.max_linear_vel = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.safety_distance = self.get_parameter('safety_distance').get_parameter_value().double_value
        self.obstacle_radius = self.get_parameter('obstacle_influence_radius').get_parameter_value().double_value
        self.camera_weight = self.get_parameter('camera_detection_weight').get_parameter_value().double_value
        self.lidar_weight = self.get_parameter('lidar_detection_weight').get_parameter_value().double_value
        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        update_rate = self.get_parameter('update_rate').get_parameter_value().double_value
        
        # Robot state (assumed at origin initially, will be updated from sensor data)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        # Obstacle storage (in robot frame)
        self.obstacles: deque = deque(maxlen=100)
        self.camera_objects: deque = deque(maxlen=50)
        
        # Last update times
        self.last_lidar_update = time.time()
        self.last_camera_update = time.time()
        self.data_timeout = 1.0  # seconds
        
        # Create subscribers
        self.lidar_subscription = self.create_subscription(
            PointStamped,
            '/lidar/obstacles',
            self.lidar_obstacle_callback,
            10
        )
        
        self.camera_subscription = self.create_subscription(
            String,
            '/camera/detections',
            self.camera_detections_callback,
            10
        )
        
        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Control timer
        timer_period = 1.0 / update_rate
        self.control_timer = self.create_timer(timer_period, self.control_loop)
        
        self.get_logger().info('Obstacle Avoidance Node started')
        self.get_logger().info(f'Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        self.get_logger().info(f'Max velocity: {self.max_linear_vel:.2f} m/s, {self.max_angular_vel:.2f} rad/s')
        self.get_logger().info(f'Safety distance: {self.safety_distance:.2f} m')
    
    def lidar_obstacle_callback(self, msg: PointStamped):
        """Callback for LIDAR obstacle messages"""
        self.last_lidar_update = time.time()
        
        # Store obstacle in robot frame (LIDAR already provides relative coordinates)
        obstacle = {
            'x': msg.point.x,
            'y': msg.point.y,
            'z': msg.point.z,
            'distance': math.sqrt(msg.point.x**2 + msg.point.y**2),
            'timestamp': time.time()
        }
        
        # Only consider obstacles within influence radius
        if obstacle['distance'] < self.obstacle_radius:
            self.obstacles.append(obstacle)
            self.get_logger().debug(f'LIDAR obstacle at ({obstacle["x"]:.2f}, {obstacle["y"]:.2f}), distance: {obstacle["distance"]:.2f}m')
    
    def camera_detections_callback(self, msg: String):
        """Callback for camera detection messages"""
        self.last_camera_update = time.time()
        
        try:
            detections = json.loads(msg.data)
            if isinstance(detections, list):
                for detection in detections:
                    # Convert camera detections to obstacle representation
                    # Camera provides bounding boxes, we'll treat them as potential obstacles
                    bbox = detection.get('bbox', {})
                    class_name = detection.get('class_name', 'unknown')
                    confidence = detection.get('confidence', 0.0)
                    
                    # Only consider high-confidence detections of obstacles
                    obstacle_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']
                    if class_name in obstacle_classes and confidence > 0.5:
                        # Estimate obstacle position (simplified: assume center of image)
                        # In real implementation, would use depth estimation or stereo vision
                        estimated_distance = 3.0  # Default assumption
                        
                        camera_obj = {
                            'class_name': class_name,
                            'confidence': confidence,
                            'estimated_distance': estimated_distance,
                            'bbox': bbox,
                            'timestamp': time.time()
                        }
                        self.camera_objects.append(camera_obj)
                        self.get_logger().debug(f'Camera detected {class_name} with confidence {confidence:.2f}')
        except json.JSONDecodeError as e:
            self.get_logger().warning(f'Failed to parse camera detections: {e}')
    
    def control_loop(self):
        """Main control loop - computes and publishes velocity commands"""
        # Check if we have recent sensor data
        current_time = time.time()
        if (current_time - self.last_lidar_update > self.data_timeout and 
            current_time - self.last_camera_update > self.data_timeout):
            self.get_logger().warning('No recent sensor data, stopping robot')
            self.publish_velocity(0.0, 0.0)
            return
        
        # Check if goal is reached
        distance_to_goal = math.sqrt(
            (self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2
        )
        
        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info('Goal reached! Stopping.')
            self.publish_velocity(0.0, 0.0)
            return
        
        # Compute desired velocity using potential field method
        linear_vel, angular_vel = self.compute_velocity_command()
        
        # Publish velocity command
        self.publish_velocity(linear_vel, angular_vel)
    
    def compute_velocity_command(self) -> tuple:
        """
        Simple obstacle avoidance - move forward, turn when obstacle detected
        
        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        # Default: move forward
        linear_vel = self.max_linear_vel * 0.3  # Slower speed
        angular_vel = 0.0
        
        # Check for obstacles in front (front 90 degrees)
        front_obstacles = []
        for obstacle in list(self.obstacles):
            obs_angle = math.atan2(obstacle['y'], obstacle['x'])
            # Check if obstacle is in front (within 45 degrees of forward)
            if abs(obs_angle) < math.pi/4 and obstacle['distance'] < self.safety_distance * 2:
                front_obstacles.append(obstacle)
        
        # If obstacle detected in front, turn
        if front_obstacles:
            linear_vel = 0.0  # Stop
            angular_vel = self.max_angular_vel * 0.5  # Turn right
            self.get_logger().info(f'Obstacle detected! Turning. Distance: {min(obs["distance"] for obs in front_obstacles):.2f}m')
        else:
            self.get_logger().debug('Path clear, moving forward')
        
        return linear_vel, angular_vel
    
    def publish_velocity(self, linear: float, angular: float):
        """Publish velocity command"""
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(angular)
        
        self.cmd_vel_publisher.publish(cmd)
    
    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update robot pose (can be called from odometry subscriber if available)"""
        self.robot_x = x
        self.robot_y = y
        self.robot_theta = theta


def main(args=None):
    rclpy.init(args=args)
    
    obstacle_avoidance = ObstacleAvoidance()
    
    try:
        rclpy.spin(obstacle_avoidance)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot before shutdown
        obstacle_avoidance.publish_velocity(0.0, 0.0)
        time.sleep(0.1)
        obstacle_avoidance.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



