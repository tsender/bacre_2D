from email.mime import base
import launch
from launch.substitutions import EnvironmentVariable
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, TextSubstitution, LaunchConfiguration
import os
import launch_ros.actions
import pathlib

def generate_launch_description():
    launch_description = []

    launch_description.append(DeclareLaunchArgument('host', default_value=''))
    launch_description.append(DeclareLaunchArgument('incoming_buffer', default_value='65536'))
    launch_description.append(DeclareLaunchArgument('socket_timeout', default_value='10'))
    launch_description.append(DeclareLaunchArgument('retry_startup_delay', default_value='5'))

    launch_description.append(DeclareLaunchArgument('fragment_timeout', default_value='600'))
    launch_description.append(DeclareLaunchArgument('delay_between_messages', default_value='0'))
    launch_description.append(DeclareLaunchArgument('max_message_size', default_value='None'))
    launch_description.append(DeclareLaunchArgument('unregister_timeout', default_value='10'))

    launch_description.append(DeclareLaunchArgument('topics_glob', default_value=''))
    launch_description.append(DeclareLaunchArgument('services_glob', default_value=''))
    launch_description.append(DeclareLaunchArgument('params_glob', default_value=''))
    launch_description.append(DeclareLaunchArgument('bson_only_mode', default_value='True'))

    base_port = 9100
    num_ports = 6
    for port in range(base_port, base_port + num_ports):
        launch_description.append(
            launch_ros.actions.Node(
                prefix=['stdbuf -o L'], # Need this with distro Dashing/Eloquent so that get_logger().info appears on the screen
                package='rosbridge_server',
                executable='rosbridge_tcp',
                name=f'rosbridge_tcp{port}',
                output='screen',
                parameters=[
                    {
                    "port": port,
                    "host": LaunchConfiguration('host'),
                    "incoming_buffer": LaunchConfiguration('incoming_buffer'),
                    "socket_timeout": LaunchConfiguration('socket_timeout'),
                    "retry_startup_delay": LaunchConfiguration('retry_startup_delay'),
                    "fragment_timeout": LaunchConfiguration('fragment_timeout'),
                    "delay_between_messages": LaunchConfiguration('delay_between_messages'),
                    "max_message_size": LaunchConfiguration('max_message_size'),
                    "unregister_timeout": LaunchConfiguration('unregister_timeout'),
                    "topics_glob": LaunchConfiguration('topics_glob'),
                    "services_glob": LaunchConfiguration('services_glob'),
                    "params_glob": LaunchConfiguration('params_glob'),
                    "bson_only_mode": LaunchConfiguration('bson_only_mode'),
                    }
                ],
            )
        )
        launch_description.append(
            launch_ros.actions.Node(
                prefix=['stdbuf -o L'], # Need this with distro Dashing/Eloquent so that get_logger().info appears on the screen
                package='rosapi',
                executable='rosapi_node',
                name=f'rosapi{port}',
                output='screen',
                parameters=[
                    {
                    "topics_glob": LaunchConfiguration('topics_glob'),
                    "services_glob": LaunchConfiguration('services_glob'),
                    "params_glob": LaunchConfiguration('params_glob'),
                    }
                ],
            )
        )
    return launch.LaunchDescription(launch_description)