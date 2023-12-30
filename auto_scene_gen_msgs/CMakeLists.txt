cmake_minimum_required(VERSION 3.5)
project(auto_scene_gen_msgs)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rosidl_default_generators REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/LandscapeDescription.msg"
  "msg/OdometryWithoutCovariance.msg"
  "msg/PhysXControl.msg"
  "msg/SceneCaptureSettings.msg"
  "msg/SceneDescription.msg"
  "msg/StatusCode.msg"
  "msg/StructuralSceneActorLayout.msg"
  "msg/VehicleNodeOperatingInfo.msg"
  "msg/VehicleStatus.msg"
)

set(srv_files
  "srv/AnalyzeScenario.srv"
  "srv/NotifyReady.srv"
  "srv/RegisterVehicleNode.srv"
  "srv/RunScenario.srv"
  # "srv/SceneCapture.srv"
  "srv/WorkerIssueNotification.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files} ${srv_files}
  DEPENDENCIES nav_msgs geometry_msgs sensor_msgs std_msgs
 )

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

ament_package()
