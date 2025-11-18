#!/usr/bin/env python3
"""
LIDAR Publisher Node
Publishes LIDAR scan data from Gazebo simulation to /lidar/scan topic
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import math


class LidarPublisher(Node):
    """Relays LIDAR scan data from Gazebo"""

    def __init__(self):
        super().__init__('lidar_publisher')
        
        # Gazebo publishes directly to /lidar/scan, so we just need to relay it
        # This node mainly serves as a monitor and can add processing if needed
        self.gazebo_lidar_received = False
        
        # Subscribe to Gazebo lidar
        self.gazebo_subscription = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_callback,
            10
        )
        
        # Timer to check if we're receiving data
        timer_period = 1.0  # 1 Hz check
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('LIDAR Publisher Node started')
        self.get_logger().info('Monitoring Gazebo LIDAR data on /lidar/scan')

    def lidar_callback(self, msg):
        """Callback for LIDAR data from Gazebo"""
        self.gazebo_lidar_received = True
        # Data is already published by Gazebo, we just monitor it
        self.get_logger().debug(f'Received LIDAR scan with {len(msg.ranges)} readings')
    
    def timer_callback(self):
        """Check if we're receiving LIDAR data"""
        if not self.gazebo_lidar_received:
            # Use standard warning logger; warn_once is not provided by rclpy logger
            self.get_logger().warning('No Gazebo LIDAR data received, check if Gazebo is running with LIDAR plugin')
        
        self.gazebo_lidar_received = False


def main(args=None):
    rclpy.init(args=args)
    
    lidar_publisher = LidarPublisher()
    
    try:
        rclpy.spin(lidar_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


