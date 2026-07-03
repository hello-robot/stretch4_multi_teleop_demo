from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'multi_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'pynput'],
    zip_safe=True,
    maintainer='eshort',
    maintainer_email='elaine.short@hello-robot.com',
    description='A library for mapping input interfaces to control schemes.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'control_scheme_example = examples.control_scheme_example:main',
            'input_interface_example = examples.input_interface_example:main',
            'orchestrator = multi_teleop.teleop_orchestrator:main',
        ],
    },
)
