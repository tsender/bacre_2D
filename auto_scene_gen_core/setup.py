import os
import glob
from setuptools import setup

package_name = 'auto_scene_gen_core'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tsender',
    maintainer_email='tsender97@gmail.com',
    description='The core package for interacting with the AutomaticSceneGeneration plugin for UE4',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ # Format: '<executable_name> = <package_name>.<file_name>:main'
        ],
    },
)
