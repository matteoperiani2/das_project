#search the directory task2
cd task2

#build the project
colcon build --symlink-install
. install/setup.bash

#to run the formation
export formation_shape=cube
ros2 launch task2 task2_formation.launch.py

#to run the collision
export formation_shape=cube
ros2 launch task2 task2_collision.launch.py

#to run the leader_follower
export formation_shape=cube
export reference=point
ros2 launch task2 task2_leader_follower.launch.py

#to run the obstacle avoidance
export formation_shape=cube
export reference=point
ros2 launch task2 task2_obstacle_avoidance.launch.py

#NB: formation_shape can be selected between cube, pentagon, octahedron and pyramid
#NB: reference can be selected between trajectory and point
