#!/usr/bin/env python3
"""
Camera Publisher Node
Relays camera images from Gazebo to /camera/image_raw topic.
If Gazebo camera is not available, generates dummy test images.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np


class CameraPublisher(Node):
    """Relays or generates camera image data"""

    def __init__(self):
        super().__init__('camera_publisher')
        
        # Create publisher for camera images
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Try multiple common Gazebo camera topics
        self.gazebo_topics = [
            '/camera/image_raw',
            '/camera/rgb/image_raw', 
            '/robot/camera/image_raw',
            '/front_camera/image_raw'
        ]
        
        # Subscribe to all possible Gazebo camera topics
        self.gazebo_subscriptions = []
        for topic in self.gazebo_topics:
            sub = self.create_subscription(
                Image,
                topic,
                self.gazebo_image_callback,
                10
            )
            self.gazebo_subscriptions.append(sub)
        
        # Fallback: Generate dummy images if Gazebo not available
        timer_period = 1.0 / 30.0  # 30 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Camera parameters for dummy images
        self.width = 640
        self.height = 480
        self.frame_id = 'camera_link'
        self.last_gazebo_image = None
        self.gazebo_image_received = False
        self.warning_logged = False
        
        self.get_logger().info('Camera Publisher Node started')
        self.get_logger().info(f'Subscribing to Gazebo camera topics: {self.gazebo_topics}')
        self.get_logger().info('Publishing to: /camera/image_raw')
        self.get_logger().info('Note: If Gazebo camera not available, will generate dummy test images')

    def gazebo_image_callback(self, msg):
        """Callback for images from Gazebo camera plugin"""
        # Relay the image directly
        self.publisher_.publish(msg)
        self.gazebo_image_received = True
        self.last_gazebo_image = msg
        self.get_logger().debug('Relayed image from Gazebo camera')

    def timer_callback(self):
        """Fallback: Generate dummy images if Gazebo camera not available"""
        # Only generate dummy if we haven't received Gazebo images recently
        if not self.gazebo_image_received:
            if not self.warning_logged:
                self.get_logger().warn('No Gazebo camera data received, generating dummy images')
                self.warning_logged = True
            self.generate_dummy_image()
        
        # Reset flag periodically to check if Gazebo is back
        self.gazebo_image_received = False
    
    def generate_dummy_image(self):
        """Generate a simple test image"""
        # Create a simple gradient image
        image_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                image_data[y, x] = [x % 256, y % 256, (x + y) % 256]
        
        # Create ROS Image message
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = self.width * 3
        msg.data = image_data.flatten().tobytes()
        
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    camera_publisher = CameraPublisher()
    
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

