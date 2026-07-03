from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # --- EDIT THESE PATHS AS NEEDED ---
    default_model_dir = "/home/eshort/human_sandbox/headtracking_demo/model"
    face_model = PathJoinSubstitution([default_model_dir, "face_landmarker.task"])
    hand_model = PathJoinSubstitution([default_model_dir, "hand_landmarker.task"])
    # ----------------------------------

    pkg_share = FindPackageShare('teleop_interfaces')

    return LaunchDescription([
        DeclareLaunchArgument('face_model_path', default_value=face_model),
        DeclareLaunchArgument('hand_model_path', default_value=hand_model),
        DeclareLaunchArgument('visualize', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),

        # 0. Camera Publisher (using usb_cam)
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
            remappings=[
                ('image_raw', LaunchConfiguration('image_topic'))
            ]
        ),

        # 1. Head and Face Tracker
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
            arguments=['-c', PathJoinSubstitution([pkg_share, 'config', 'head_face_config.yaml'])]
        ),

        # 2. Dual Hands Tracker
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
            arguments=['-c', PathJoinSubstitution([pkg_share, 'config', 'hands_config.yaml'])]
        ),

        # 3. Single Hand Tracker (tracking right hand by default in this launch)
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
            arguments=['-c', PathJoinSubstitution([pkg_share, 'config', 'hand_config.yaml'])]
        ),
    ])
