cmake_minimum_required(VERSION 3.5)
project(auto_scene_gen_core)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(auto_scene_gen_msgs REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

if(BUILD_TESTING)
find_package(ament_lint_auto REQUIRED)
# the following line skips the linter which checks for copyrights
# uncomment the line when a copyright and license is not present in all source files
#set(ament_cmake_copyright_FOUND TRUE)
# the following line skips cpplint (only works in a git repo)
# uncomment the line when this package is not in a git repo
#set(ament_cmake_cpplint_FOUND TRUE)
ament_lint_auto_find_test_dependencies()
endif()

# Add include directories to targets
include_directories(include)

# Create library
add_library(${PROJECT_NAME} SHARED src/vehicle_node.cpp)
ament_target_dependencies(${PROJECT_NAME} rclcpp auto_scene_gen_msgs)
target_link_libraries(${PROJECT_NAME} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
# ament_export_dependencies(${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
ament_export_libraries(${PROJECT_NAME})

# Install include
install(
  DIRECTORY include/
  DESTINATION include
)

# Install everything else
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

ament_package()
