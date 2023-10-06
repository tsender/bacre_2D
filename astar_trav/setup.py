import os
import glob
from setuptools import setup

package_name = 'astar_trav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob.glob('launch/*_launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tsender',
    maintainer_email='tsender97@gmail.com',
    description='Package for an traversability-based A* vehicle model',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ # Format: '<executable_name> = <package_name>.<file_name>:main'
            'astar_trav_2D_all_in_one = astar_trav.astar_trav_2D_all_in_one_node:main',
            'astar_trav_2D_mapper = astar_trav.astar_trav_2D_mapping_node:main',
            'astar_trav_2D_planner = astar_trav.astar_trav_2D_planning_node:main',
            'simple_path_follower = astar_trav.simple_path_follower_node:main',
        ],
    },
)
