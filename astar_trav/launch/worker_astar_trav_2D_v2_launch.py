import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, TextSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros.actions

# Use this to launch additional A* trav nodes
def generate_launch_description():
    launch_description = []

    # Vehicle node params
    launch_description.append(DeclareLaunchArgument("vehicle_name", default_value="vehicle"))
    launch_description.append(DeclareLaunchArgument("wid", default_value="0"))
    launch_description.append(DeclareLaunchArgument("asg_client_name", default_value="asg_client"))
    launch_description.append(DeclareLaunchArgument("asg_client_ip_addr", default_value=""))
    launch_description.append(DeclareLaunchArgument("ssh_username", default_value=""))
    launch_description.append(DeclareLaunchArgument("debug_mode", default_value="False"))

    # Add A* trav nodes
    launch_description.append(
        launch_ros.actions.Node(
            prefix=["stdbuf -o L"], # Need this with distro Dashing so that get_logger().info appears on the screen
            package="astar_trav", 
            executable="astar_trav_2D_mapper", # Defined in the setup.py entrypoints
            name=["astar_trav_2D_mapper", LaunchConfiguration("wid")], # This will change the node's name
            output="screen",
            parameters=[
                {"vehicle_name": LaunchConfiguration("vehicle_name"),
                "wid": LaunchConfiguration("wid"),
                "asg_client_name": LaunchConfiguration("asg_client_name"),
                "asg_client_ip_addr": LaunchConfiguration("asg_client_ip_addr"),
                "ssh_username": LaunchConfiguration("ssh_username"),
                "debug_mode": LaunchConfiguration("debug_mode")
                }
            ]
        )
    )
    launch_description.append(
        launch_ros.actions.Node(
            prefix=["stdbuf -o L"], # Need this with distro Dashing so that get_logger().info appears on the screen
            package="astar_trav", 
            executable="astar_trav_2D_planner", # Defined in the setup.py entrypoints
            name=["astar_trav_2D_planner", LaunchConfiguration("wid")], # This will change the node's name
            output="screen",
            parameters=[
                {"vehicle_name": LaunchConfiguration("vehicle_name"),
                "wid": LaunchConfiguration("wid"),
                "asg_client_name": LaunchConfiguration("asg_client_name"),
                "asg_client_ip_addr": LaunchConfiguration("asg_client_ip_addr"),
                "ssh_username": LaunchConfiguration("ssh_username"),
                "debug_mode": LaunchConfiguration("debug_mode")
                }
            ]
        )
    )
    launch_description.append(
        launch_ros.actions.Node(
            prefix=["stdbuf -o L"], # Need this with distro Dashing so that get_logger().info appears on the screen
            package="astar_trav", 
            executable="simple_path_follower", # Defined in the setup.py entrypoints
            name=["simple_path_follower", LaunchConfiguration("wid")], # This will change the node's name
            output="screen",
            parameters=[
                {"vehicle_name": LaunchConfiguration("vehicle_name"),
                "wid": LaunchConfiguration("wid"),
                "asg_client_name": LaunchConfiguration("asg_client_name"),
                "asg_client_ip_addr": LaunchConfiguration("asg_client_ip_addr"),
                "ssh_username": LaunchConfiguration("ssh_username"),
                "debug_mode": LaunchConfiguration("debug_mode")
                }
            ]
        )
    )
    # launch_description.append(
    #     launch_ros.actions.Node(
    #         prefix=["stdbuf -o L"], # Need this with distro Dashing so that get_logger().info appears on the screen
    #         package="test_vehicle_cpp", 
    #         executable="test_vehicle_node", # Defined in the setup.py entrypoints
    #         name=["test_vehicle_node", LaunchConfiguration("wid")], # This will change the node's name
    #         output="screen",
    #         parameters=[
    #             {"vehicle_name": LaunchConfiguration("vehicle_name"),
    #             "wid": LaunchConfiguration("wid"),
    #             "asg_client_name": LaunchConfiguration("asg_client_name"),
    #             "asg_client_ip_addr": LaunchConfiguration("asg_client_ip_addr"),
    #             "ssh_username": LaunchConfiguration("ssh_username"),
    #             "debug_mode": LaunchConfiguration("debug_mode")
    #             }
    #         ]
    #     )
    # )

    return launch.LaunchDescription(launch_description)