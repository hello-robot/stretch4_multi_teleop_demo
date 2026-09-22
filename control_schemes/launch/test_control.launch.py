"""Launches sim_direct_position_control (MuJoCo sim) and teleop_interfaces' gui_node,
wired to feed the sim node's input topic.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    control_schemes_dir = get_package_share_directory('control_schemes')
    teleop_interfaces_dir = get_package_share_directory('teleop_interfaces')
    
    gui_config = os.path.join(control_schemes_dir, 'config', 'test_gui.yaml')
    
    return LaunchDescription([
        Node(
            package='control_schemes',
            executable='sim_direct_position_control',
            name='sim_direct_position_control',
            output='screen'
        ),
        Node(
            package='teleop_interfaces',
            executable='gui_node',
            name='gui_node',
            output='screen',
            parameters=[{'config_file': gui_config}],
            remappings=[
                ('gui_node/output', 'sim_direct_position_control/input')
            ]
        )
    ])
