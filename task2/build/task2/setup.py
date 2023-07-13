from setuptools import setup
from glob import glob

package_name = 'task2'
scripts = ['the_agent_formation', 'the_agent_collision', 'the_agent_leader_follower', 'the_agent_complete', 'the_agent_obstacle_avoidance', 'visualizer', 'obstacle_visualizer', 'goal_visualizer']

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*_formation.launch.py')),
        ('share/' + package_name, glob('launch/*_leader_follower.launch.py')),
        ('share/' + package_name, glob('launch/*_collision.launch.py')),
        ('share/' + package_name, glob('launch/*_complete.launch.py')),
        ('share/' + package_name, glob('launch/*_obstacle_avoidance.launch.py')),
        ('share/' + package_name, glob('resource/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luca',
    maintainer_email='luca@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            '{1} = {0}.{1}:main'.format(package_name, script) for script in scripts
        ],
    },
)
