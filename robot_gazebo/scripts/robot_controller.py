#!/usr/bin/env python3
"""
Robot Controller Node
Provides keyboard control and autonomous navigation for the robot
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import sys
import select
import termios
import tty


class RobotController(Node):
    """Controls robot movement via keyboard or autonomous navigation"""

    def __init__(self):
        super().__init__('robot_controller')
        
        # Create publisher for robot movement
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe to LIDAR for obstacle avoidance
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_callback,
            10
        )
        
        # Robot parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 1.0  # rad/s
        
        # Obstacle avoidance parameters
        self.obstacle_distance = 1.0  # Stop if obstacle within 1m
        self.latest_scan = None
        self.autonomous_mode = False
        
        # Timer for autonomous navigation
        self.timer = self.create_timer(0.1, self.autonomous_callback)
        
        self.get_logger().info('Robot Controller started')
        self.get_logger().info('Controls:')
        self.get_logger().info('  w/s: forward/backward')
        self.get_logger().info('  a/d: turn left/right')
        self.get_logger().info('  x: stop')
        self.get_logger().info('  q: quit')
        self.get_logger().info('  t: toggle autonomous mode')
        
        # Setup terminal for keyboard input
        self.settings = termios.tcgetattr(sys.stdin)

    def lidar_callback(self, msg):
        """Store latest LIDAR scan for obstacle avoidance"""
        self.latest_scan = msg

    def autonomous_callback(self):
        """Autonomous navigation using LIDAR data"""
        if not self.autonomous_mode or self.latest_scan is None:
            return
        
        # Simple obstacle avoidance algorithm
        ranges = self.latest_scan.ranges
        if not ranges:
            return
        
        # Check front sectors for obstacles
        front_ranges = ranges[:30] + ranges[-30:]  # Front 60 degrees
        min_front_distance = min(front_ranges) if front_ranges else float('inf')
        
        twist = Twist()
        
        if min_front_distance > self.obstacle_distance:
            # Path clear, move forward
            twist.linear.x = self.linear_speed * 0.5
            twist.angular.z = 0.0
        else:
            # Obstacle detected, turn right
            twist.linear.x = 0.0
            twist.angular.z = -self.angular_speed * 0.5
        
        self.cmd_vel_publisher.publish(twist)

    def get_key(self):
        """Get keyboard input"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        """Main control loop"""
        try:
            while rclpy.ok():
                key = self.get_key()
                twist = Twist()
                
                if key == 'w':
                    twist.linear.x = self.linear_speed
                    self.autonomous_mode = False
                elif key == 's':
                    twist.linear.x = -self.linear_speed
                    self.autonomous_mode = False
                elif key == 'a':
                    twist.angular.z = self.angular_speed
                    self.autonomous_mode = False
                elif key == 'd':
                    twist.angular.z = -self.angular_speed
                    self.autonomous_mode = False
                elif key == 'x':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.autonomous_mode = False
                elif key == 't':
                    self.autonomous_mode = not self.autonomous_mode
                    mode = "ON" if self.autonomous_mode else "OFF"
                    self.get_logger().info(f'Autonomous mode: {mode}')
                    continue
                elif key == 'q':
                    break
                else:
                    continue
                
                if not self.autonomous_mode:
                    self.cmd_vel_publisher.publish(twist)
                
                rclpy.spin_once(self, timeout_sec=0.01)
                
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
        finally:
            # Stop robot
            twist = Twist()
            self.cmd_vel_publisher.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    
    controller = RobotController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()