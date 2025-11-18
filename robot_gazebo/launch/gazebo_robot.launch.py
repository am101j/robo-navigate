#!/usr/bin/env python3
"""
Complete launch file to start Gazebo and spawn the virtual robot
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Package directories
    pkg_share = get_package_share_directory('robot_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    
    # URDF file path
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro')
    
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_file = LaunchConfiguration('world_file')
    
    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_world_file_cmd = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(pkg_share, 'worlds', 'test_world.world'),
        description='Path to world file (optional)'
    )
    
    # Start Gazebo with proper ROS integration
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )
    
    # Process URDF with xacro
    robot_description_content = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )
    
    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': robot_description_content}
        ]
    )
    
    # Spawn robot in Gazebo (with delay to ensure Gazebo is ready)
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'virtual_robot',
            # Read the robot description from the ROS parameter set by robot_state_publisher
            '-param', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.2'
        ],
        output='screen'
    )
    
    # --- Your Custom Application Nodes ---
    
    # C++ LIDAR Obstacle Detector
    lidar_detector_node = Node(
        package='robot_gazebo',
        executable='lidar_obstacle_detector',
        name='lidar_obstacle_detector',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Python Robot Controller
    robot_controller_node = Node(
        package='robot_gazebo',
        executable='robot_controller.py',
        name='robot_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Python Obstacle Avoidance
    obstacle_avoidance_node = Node(
        package='robot_gazebo',
        executable='obstacle_avoidance.py',
        name='obstacle_avoidance',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Python Object Detector
    object_detector_node = Node(
        package='robot_gazebo',
        executable='object_detector.py',
        name='object_detector',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Create launch description
    ld = LaunchDescription()
    
    # Add declared launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_file_cmd)
    
    # Start Gazebo first
    ld.add_action(gazebo)
    
    # Start robot state publisher immediately
    ld.add_action(robot_state_publisher_node)
    
    # Delay spawn entity to ensure Gazebo is ready (4 seconds) and spawn slightly higher
    # to avoid the model intersecting the ground on spawn.
    spawn_entity_node.arguments[spawn_entity_node.arguments.index('-z') + 1] = '0.5'
    ld.add_action(TimerAction(
        period=4.0,
        actions=[spawn_entity_node]
    ))

    # Delay application nodes to ensure robot is spawned and stable (6 seconds)
    ld.add_action(TimerAction(
        period=6.0,
        actions=[
            lidar_detector_node,
            robot_controller_node,
            obstacle_avoidance_node,
            object_detector_node
        ]
    ))
    
    return ld