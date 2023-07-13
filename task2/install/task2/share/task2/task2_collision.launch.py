from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
import numpy as np


Tmax = 5.0         # simulation time
n_p = 3		        # dimension of x_i 

visu_frequency = 100        # [Hz]
dt = 1/visu_frequency       # sampling period
horizon = np.arange(0.0, Tmax, dt)
TT = len(horizon)


def generate_launch_description():

    formation_shape = os.environ['formation_shape']

    if formation_shape == 'pentagon':
        NN = 5
        print(f'SHAPE: {formation_shape} AGENTS: {NN}')

    if formation_shape == 'pyramid':
        NN = 5
        print(f'SHAPE: {formation_shape} AGENTS: {NN}')

    if formation_shape == 'octahedron':
        NN = 6
        print(f'SHAPE: {formation_shape} AGENTS: {NN}')

    if formation_shape == 'cube':
        NN = 8
        print(f'SHAPE: {formation_shape} AGENTS: {NN}')

    launch_description = [] # Append here your nodes

    ################################################################################
    # RVIZ
    ################################################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('task2')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    launch_description.append(
        Node(
            package='rviz2', 
            executable='rviz2', 
            arguments=['-d', rviz_config_file],
            # output='screen',
            # prefix='xterm -title "rviz2" -hold -e'
            ))

    ################################################################################

    for ii in range(NN):
        index_ii =  ii*n_p + np.arange(n_p)	

        launch_description.append(
            Node(
                package='task2',
                namespace =f'agent_{ii}',
                executable='the_agent_collision',
                parameters=[{ # dictionary
                                'agent_id': ii, 
                                'n_p': n_p,
                                'sampling period': dt,
                                'max_iters': TT
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{ii}" -hold -e',
            ))
        launch_description.append(
            Node(
                package='task2', 
                namespace='agent_{}'.format(ii),
                executable='visualizer', 
                parameters=[{
                                'agent_id': ii,
                                'index_ii': list(map(float, index_ii)),
                                'node_frequency': visu_frequency,
                                }],
            ))


    return LaunchDescription(launch_description)