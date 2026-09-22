"""Launches direct position control, the space mouse, and the hand tracker.

Note: ``sim_direct_position_control`` is launched here; the orchestrator
itself is not launched here.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    teleop_interfaces_dir = get_package_share_directory('teleop_interfaces')
    
    hand_config = os.path.join(teleop_interfaces_dir, 'config', 'hand_config.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='path/to/hand_landmarker.task',
            description='Path to the Mediapipe hand landmarker model'
        ),

        # Direct Position Control Node (Simulator + ROS Interface)
        Node(
            package='control_schemes',
            executable='sim_direct_position_control',
            name='sim_direct_position_control',
            output='screen'
        ),
        
        # Space Mouse Node
        Node(
            package='teleop_interfaces',
            executable='space_mouse_node',
            name='space_mouse_node',
            output='screen'
        ),
        
        # Hand Tracker Node
        Node(
            package='teleop_interfaces',
            executable='hand_tracker',
            name='hand_tracker',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'hand_to_track': 'right'
            }],
            arguments=['-c', hand_config]
        )
    ])
