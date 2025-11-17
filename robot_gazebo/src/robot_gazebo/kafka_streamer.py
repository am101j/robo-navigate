#!/usr/bin/env python3
"""
Kafka Streaming Node
Streams ROS2 messages from /camera/detections and /lidar/obstacles to Kafka topics.
Includes error handling and JSON serialization.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import json
import time
from typing import Optional

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, KafkaTimeoutError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Install with: pip install kafka-python")


class KafkaStreamer(Node):
    """ROS2 node that streams messages to Kafka topics"""

    def __init__(self):
        super().__init__('kafka_streamer')
        
        # Check if Kafka is available
        if not KAFKA_AVAILABLE:
            self.get_logger().error('kafka-python library not available!')
            self.get_logger().error('Install with: pip install kafka-python')
            return
        
        # Declare parameters
        self.declare_parameter('kafka_bootstrap_servers', 'localhost:9092')
        self.declare_parameter('kafka_camera_topic', 'camera_detections')
        self.declare_parameter('kafka_lidar_topic', 'lidar_obstacles')
        self.declare_parameter('kafka_client_id', 'ros2_kafka_streamer')
        self.declare_parameter('kafka_acks', 'all')
        self.declare_parameter('kafka_retries', 3)
        self.declare_parameter('kafka_max_in_flight_requests_per_connection', 5)
        self.declare_parameter('kafka_compression_type', 'gzip')
        self.declare_parameter('kafka_request_timeout_ms', 30000)
        
        # Get parameters
        bootstrap_servers = self.get_parameter('kafka_bootstrap_servers').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('kafka_camera_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('kafka_lidar_topic').get_parameter_value().string_value
        client_id = self.get_parameter('kafka_client_id').get_parameter_value().string_value
        acks = self.get_parameter('kafka_acks').get_parameter_value().string_value
        retries = self.get_parameter('kafka_retries').get_parameter_value().integer_value
        max_in_flight = self.get_parameter('kafka_max_in_flight_requests_per_connection').get_parameter_value().integer_value
        compression_type = self.get_parameter('kafka_compression_type').get_parameter_value().string_value
        request_timeout = self.get_parameter('kafka_request_timeout_ms').get_parameter_value().integer_value
        
        # Initialize Kafka producer
        self.producer: Optional[KafkaProducer] = None
        self.kafka_connected = False
        self.connection_retries = 0
        self.max_connection_retries = 5
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                client_id=client_id,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks=acks,
                retries=retries,
                max_in_flight_requests_per_connection=max_in_flight,
                compression_type=compression_type,
                request_timeout_ms=request_timeout,
                api_version=(0, 10, 1)  # Use a stable API version
            )
            self.kafka_connected = True
            self.get_logger().info(f'Kafka producer initialized successfully')
            self.get_logger().info(f'Bootstrap servers: {bootstrap_servers}')
            self.get_logger().info(f'Camera topic: {self.camera_topic}')
            self.get_logger().info(f'LIDAR topic: {self.lidar_topic}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Kafka producer: {str(e)}')
            self.get_logger().warn('Node will continue but messages will not be streamed to Kafka')
            self.kafka_connected = False
        
        # Statistics
        self.camera_messages_sent = 0
        self.camera_messages_failed = 0
        self.lidar_messages_sent = 0
        self.lidar_messages_failed = 0
        
        # Create subscribers
        self.camera_subscription = self.create_subscription(
            String,
            '/camera/detections',
            self.camera_detections_callback,
            10
        )
        
        self.lidar_subscription = self.create_subscription(
            PointStamped,
            '/lidar/obstacles',
            self.lidar_obstacles_callback,
            10
        )
        
        # Timer for statistics logging
        self.stats_timer = self.create_timer(10.0, self.log_statistics)
        
        self.get_logger().info('Kafka Streamer Node started')
        self.get_logger().info('Subscribing to: /camera/detections')
        self.get_logger().info('Subscribing to: /lidar/obstacles')
    
    def camera_detections_callback(self, msg: String):
        """Callback for camera detection messages"""
        if not self.kafka_connected or not self.producer:
            return
        
        try:
            # Parse the JSON string from the message
            detections_data = json.loads(msg.data)
            
            # Create message payload with metadata
            # Note: std_msgs/String doesn't have a header, so we use current time
            current_time = self.get_clock().now()
            time_msg = current_time.to_msg()
            payload = {
                'timestamp': time.time(),
                'ros_timestamp': {
                    'sec': time_msg.sec,
                    'nanosec': time_msg.nanosec
                },
                'frame_id': 'camera',
                'source': 'camera',
                'detections': detections_data
            }
            
            # Send to Kafka
            future = self.producer.send(
                self.camera_topic,
                value=payload,
                key='camera_detections'
            )
            
            # Add callback for success/failure
            future.add_callback(self.on_send_success)
            future.add_errback(self.on_send_error)
            
            self.camera_messages_sent += 1
            self.get_logger().debug(f'Sent camera detection to Kafka topic: {self.camera_topic}')
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse camera detection JSON: {str(e)}')
            self.camera_messages_failed += 1
        except Exception as e:
            self.get_logger().error(f'Error processing camera detection: {str(e)}')
            self.camera_messages_failed += 1
    
    def lidar_obstacles_callback(self, msg: PointStamped):
        """Callback for LIDAR obstacle messages"""
        if not self.kafka_connected or not self.producer:
            return
        
        try:
            # Create message payload with metadata
            payload = {
                'timestamp': time.time(),
                'ros_timestamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                },
                'frame_id': msg.header.frame_id,
                'source': 'lidar',
                'obstacle': {
                    'x': float(msg.point.x),
                    'y': float(msg.point.y),
                    'z': float(msg.point.z)
                }
            }
            
            # Send to Kafka
            future = self.producer.send(
                self.lidar_topic,
                value=payload,
                key='lidar_obstacle'
            )
            
            # Add callback for success/failure
            future.add_callback(self.on_send_success)
            future.add_errback(self.on_send_error)
            
            self.lidar_messages_sent += 1
            self.get_logger().debug(f'Sent LIDAR obstacle to Kafka topic: {self.lidar_topic}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing LIDAR obstacle: {str(e)}')
            self.lidar_messages_failed += 1
    
    def on_send_success(self, record_metadata):
        """Callback for successful Kafka send"""
        self.get_logger().debug(
            f'Message sent successfully to topic={record_metadata.topic}, '
            f'partition={record_metadata.partition}, offset={record_metadata.offset}'
        )
    
    def on_send_error(self, exception):
        """Callback for failed Kafka send"""
        self.get_logger().error(f'Failed to send message to Kafka: {str(exception)}')
        
        # Check if it's a connection error
        if isinstance(exception, (KafkaError, KafkaTimeoutError)):
            self.get_logger().warn('Kafka connection error detected')
            self.kafka_connected = False
            self.connection_retries += 1
            
            if self.connection_retries < self.max_connection_retries:
                self.get_logger().info(f'Retrying Kafka connection (attempt {self.connection_retries}/{self.max_connection_retries})')
                # Attempt to reconnect
                self.reconnect_kafka()
    
    def reconnect_kafka(self):
        """Attempt to reconnect to Kafka"""
        try:
            if self.producer:
                self.producer.close()
            
            bootstrap_servers = self.get_parameter('kafka_bootstrap_servers').get_parameter_value().string_value
            client_id = self.get_parameter('kafka_client_id').get_parameter_value().string_value
            acks = self.get_parameter('kafka_acks').get_parameter_value().string_value
            retries = self.get_parameter('kafka_retries').get_parameter_value().integer_value
            max_in_flight = self.get_parameter('kafka_max_in_flight_requests_per_connection').get_parameter_value().integer_value
            compression_type = self.get_parameter('kafka_compression_type').get_parameter_value().string_value
            request_timeout = self.get_parameter('kafka_request_timeout_ms').get_parameter_value().integer_value
            
            self.producer = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                client_id=client_id,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks=acks,
                retries=retries,
                max_in_flight_requests_per_connection=max_in_flight,
                compression_type=compression_type,
                request_timeout_ms=request_timeout,
                api_version=(0, 10, 1)
            )
            
            self.kafka_connected = True
            self.connection_retries = 0
            self.get_logger().info('Successfully reconnected to Kafka')
            
        except Exception as e:
            self.get_logger().error(f'Failed to reconnect to Kafka: {str(e)}')
            self.kafka_connected = False
    
    def log_statistics(self):
        """Log statistics about message streaming"""
        if self.kafka_connected:
            self.get_logger().info(
                f'Statistics - Camera: {self.camera_messages_sent} sent, '
                f'{self.camera_messages_failed} failed | '
                f'LIDAR: {self.lidar_messages_sent} sent, '
                f'{self.lidar_messages_failed} failed'
            )
        else:
            self.get_logger().warn('Kafka not connected - messages are not being streamed')
    
    def destroy_node(self):
        """Cleanup on node destruction"""
        if self.producer:
            try:
                # Flush any pending messages
                self.producer.flush(timeout=5.0)
                self.producer.close(timeout=5.0)
                self.get_logger().info('Kafka producer closed successfully')
            except Exception as e:
                self.get_logger().error(f'Error closing Kafka producer: {str(e)}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    kafka_streamer = KafkaStreamer()
    
    try:
        rclpy.spin(kafka_streamer)
    except KeyboardInterrupt:
        pass
    finally:
        kafka_streamer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

