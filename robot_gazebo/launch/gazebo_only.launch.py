#!/usr/bin/env python3
"""
Launch file to start only Gazebo (for testing)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package directories
    gazebo_ros_share = FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    
    # Launch configuration variables
    world_file = LaunchConfiguration('world_file')
    
    # Declare launch arguments
    declare_world_file_cmd = DeclareLaunchArgument(
        'world_file',
        default_value='',
        description='Path to world file (optional)'
    )
    
    # Start Gazebo server
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([gazebo_ros_share, 'launch', 'gzserver.launch.py'])
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )
    
    # Start Gazebo client
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([gazebo_ros_share, 'launch', 'gzclient.launch.py'])
        ])
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add declared launch arguments
    ld.add_action(declare_world_file_cmd)
    
    # Start Gazebo
    ld.add_action(gazebo_server)
    ld.add_action(gazebo_client)
    
    return ld