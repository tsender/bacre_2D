import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, TextSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros.actions

# Dashing / Eloquent: use node_executable and node_name
# Foxy+: use executable and name

def generate_launch_description():
    launch_description = []

    launch_description.append(
        launch_ros.actions.Node(
            prefix=["stdbuf -o L"], # Need this with distro Dashing so that logger.info appears on the screen
            package="adversarial_scene_gen", 
            executable="bacre_2D_astar_trav", # Defined in the setup.py entrypoints
            name="bacre_2D_astar_trav", # This will change the node's name
            output="screen",
        )
    )

    return launch.LaunchDescription(launch_description)