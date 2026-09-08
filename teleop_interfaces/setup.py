from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'teleop_interfaces'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'pyspacemouse', 'pynput', 'mediapipe', 'opencv-python', 'evdev'],
    zip_safe=True,
    maintainer='eshort',
    maintainer_email='elaine.short@hello-robot.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'space_mouse_node = teleop_interfaces.space_mouse_node:main',
            'mouse_node = teleop_interfaces.mouse_node:main',
            'gamepad_node = teleop_interfaces.gamepad_node:main',
            'head_face_tracker = teleop_interfaces.head_face_tracker:main',
            'hands_tracker = teleop_interfaces.hands_tracker:main',
            'hand_tracker = teleop_interfaces.hand_tracker:main',
            'voice_node = teleop_interfaces.voice_node:main',
            'gui_node = teleop_interfaces.gui_node:main',
            'interface_monitor = teleop_interfaces.interface_monitor:main',
        ],
    },
)
