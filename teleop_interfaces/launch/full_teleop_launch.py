"""Launches all teleop_interfaces input nodes.

Nodes: includes cam_trackers_launch.py (camera + MediaPipe trackers), plus
space_mouse_node, mouse_node, gamepad_node, voice_node, gui_node. Does not
launch interface_monitor (run separately for debugging).
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Build the full teleop_interfaces launch description."""
    # --- Models and Configs ---
    model_dir_arg = DeclareLaunchArgument(
        'model_dir',
        default_value='models',
        description='Directory containing Mediapipe task models'
    )
    
    model_dir = LaunchConfiguration('model_dir')
    face_model = PathJoinSubstitution([model_dir, "face_landmarker.task"])
    hand_model = PathJoinSubstitution([model_dir, "hand_landmarker.task"])
    
    pkg_share = FindPackageShare('teleop_interfaces')
    
    voice_config = PathJoinSubstitution([pkg_share, 'config', 'voice_config.yaml'])
    gui_config = PathJoinSubstitution([pkg_share, 'config', 'gui_config.yaml'])

    return LaunchDescription([
        # --- Launch Arguments ---
        model_dir_arg,
        DeclareLaunchArgument('face_model_path', default_value=face_model),
        DeclareLaunchArgument('hand_model_path', default_value=hand_model),
        DeclareLaunchArgument('visualize', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('voice_model_size', default_value='base.en'),

        # 1. Camera and Vision Trackers (included from cam_trackers_launch.py)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_share, 'launch', 'cam_trackers_launch.py'])
            ),
            launch_arguments={
                'model_dir': LaunchConfiguration('model_dir'),
                'face_model_path': LaunchConfiguration('face_model_path'),
                'hand_model_path': LaunchConfiguration('hand_model_path'),
                'visualize': LaunchConfiguration('visualize'),
                'image_topic': LaunchConfiguration('image_topic'),
            }.items()
        ),

        # 2. Physical Controller Nodes
        Node(
            package='teleop_interfaces',
            executable='space_mouse_node',
            name='space_mouse_node'
        ),
        Node(
            package='teleop_interfaces',
            executable='mouse_node',
            name='mouse_node'
        ),
        Node(
            package='teleop_interfaces',
            executable='gamepad_node',
            name='gamepad_node'
        ),

        # 3. Intelligent Interface Nodes
        Node(
            package='teleop_interfaces',
            executable='voice_node',
            name='voice_node',
            parameters=[{
                'model_size': LaunchConfiguration('voice_model_size')
            }],
            arguments=['-c', voice_config]
        ),
        Node(
            package='teleop_interfaces',
            executable='gui_node',
            name='gui_node',
            arguments=['-c', gui_config]
        ),
    ])
