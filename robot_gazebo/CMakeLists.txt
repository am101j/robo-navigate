cmake_minimum_required(VERSION 3.8)
project(robot_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)
find_package(robot_state_publisher REQUIRED)
# find_package(joint_state_publisher REQUIRED)  # Python package, not CMake
find_package(xacro REQUIRED)

# Build C++ executable
add_executable(lidar_obstacle_detector src/lidar_obstacle_detector.cpp)
ament_target_dependencies(lidar_obstacle_detector
  rclcpp
  sensor_msgs
  geometry_msgs
)

# Install C++ executable
install(TARGETS
  lidar_obstacle_detector
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
install(PROGRAMS
  src/robot_gazebo/camera_publisher.py
  src/robot_gazebo/lidar_publisher.py
  src/robot_gazebo/robot_controller.py
  src/robot_gazebo/object_detector.py
  src/robot_gazebo/kafka_streamer.py
  src/robot_gazebo/kafka_visualizer.py
  src/robot_gazebo/obstacle_avoidance.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install URDF and SDF files
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

# Install worlds directory if needed
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

