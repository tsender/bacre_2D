import os
import glob
from setuptools import setup

package_name = 'adversarial_scene_gen'

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
    description='Adversarial scene generation',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ # Format: '<executable_name> = <package_name>.<file_name>:main'
            f"bacre_2D_astar_trav = {package_name}.bacre_2D_astar_trav_node:main",
        ],
    },
)
