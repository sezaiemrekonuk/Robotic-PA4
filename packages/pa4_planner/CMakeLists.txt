cmake_minimum_required(VERSION 3.0.2)
project(pa4_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  duckietown_msgs
)

# Install the pa4_planner Python package (src/pa4_planner/)
catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    duckietown_msgs
)

# Make the main node executable
catkin_install_python(
  PROGRAMS src/pa4_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files and config
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
