from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # --- Paths for Models and Configs ---
    default_model_dir = "/home/eshort/human_sandbox/headtracking_demo/model"
    face_model = PathJoinSubstitution([default_model_dir, "face_landmarker.task"])
    hand_model = PathJoinSubstitution([default_model_dir, "hand_landmarker.task"])
    
    pkg_share = FindPackageShare('teleop_interfaces')
    
    voice_config = PathJoinSubstitution([pkg_share, 'config', 'voice_config.yaml'])
    gui_config = PathJoinSubstitution([pkg_share, 'config', 'gui_config.yaml'])
    head_config = PathJoinSubstitution([pkg_share, 'config', 'head_face_config.yaml'])
    hands_config = PathJoinSubstitution([pkg_share, 'config', 'hands_config.yaml'])
    hand_config = PathJoinSubstitution([pkg_share, 'config', 'hand_config.yaml'])

    return LaunchDescription([
        # --- Launch Arguments ---
        DeclareLaunchArgument('face_model_path', default_value=face_model),
        DeclareLaunchArgument('hand_model_path', default_value=hand_model),
        DeclareLaunchArgument('visualize', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('voice_model_size', default_value='base.en'),

        # 0. Camera Publisher
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device': '/dev/video0',
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'mjpeg2rgb',
                'camera_frame_id': 'camera_link',
                'io_method': 'mmap',
            }],
            remappings=[('image_raw', LaunchConfiguration('image_topic'))]
        ),

        # 1. Vision Trackers
        Node(
            package='teleop_interfaces',
            executable='head_face_tracker',
            name='head_face_tracker',
            parameters=[{
                'model_path': LaunchConfiguration('face_model_path'),
                'visualize': LaunchConfiguration('visualize'),
                'use_webcam': False,
                'image_topic': LaunchConfiguration('image_topic'),
            }],
            arguments=['-c', head_config]
        ),
        Node(
            package='teleop_interfaces',
            executable='hands_tracker',
            name='hands_tracker',
            parameters=[{
                'model_path': LaunchConfiguration('hand_model_path'),
                'visualize': LaunchConfiguration('visualize'),
                'use_webcam': False,
                'image_topic': LaunchConfiguration('image_topic'),
            }],
            arguments=['-c', hands_config]
        ),
        Node(
            package='teleop_interfaces',
            executable='hand_tracker',
            name='hand_tracker',
            parameters=[{
                'model_path': LaunchConfiguration('hand_model_path'),
                'visualize': LaunchConfiguration('visualize'),
                'hand_to_track': 'right',
                'use_webcam': False,
                'image_topic': LaunchConfiguration('image_topic'),
            }],
            arguments=['-c', hand_config]
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
