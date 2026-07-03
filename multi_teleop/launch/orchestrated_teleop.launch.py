import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    teleop_interfaces_dir = get_package_share_directory('teleop_interfaces')
    
    hand_config = os.path.join(teleop_interfaces_dir, 'config', 'hand_config.yaml')
    
    return LaunchDescription([
        # Position Control Node (Simulator + ROS Interface)
        Node(
            package='control_schemes',
            executable='position_control_node',
            name='position_control_node',
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
                'model_path': '/home/eshort/human_sandbox/headtracking_demo/model/hand_landmarker.task',
                'hand_to_track': 'right'
            }],
            arguments=['-c', hand_config]
        )
    ])
