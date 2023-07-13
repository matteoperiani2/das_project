
from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np
import os
np.random.seed(10)


class Agent(Node):

    def __init__(self):
        super().__init__('agent',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
        
        # Get the formation shape from terminal
        formation_shape = os.environ['formation_shape']
            
        # Get parameters from launch file
        self.agent_id = self.get_parameter('agent_id').value
        self.n_p = self.get_parameter('n_p').value
        self.dt = self.get_parameter('sampling period').value
        self.kk = 0
        self.max_iters = self.get_parameter('max_iters').value
        self.all_received = True

        if formation_shape == 'pentagon':
            NN = 5
            L = 1
            D = ((np.sqrt(5)+1)/2)*L 

            distances = [[0,    L,      L,    0,    D],
                        [L,     0,      D,    L,    D],
                        [L,     D,      0,    D,    L],     
                        [0,     L,      D,    0,    L],         
                        [D,     D,      L,    L,    0]]
        
        if formation_shape == 'pyramid':
            NN = 5
            L = 2
            H = 3
            D = L*np.sqrt(2)
            l = np.sqrt((H**2)+(L**2)/2)

            distances = [[0,    L,      D,    L,    l],
                        [L,     0,      L,    D,    l],
                        [D,     L,      0,    L,    l],     
                        [L,     D,      L,    0,    l],       
                        [l,     l,      l,    l,    0]]
            
        if formation_shape == 'octahedron':
            NN = 6
            L = 1
            D = L*np.sqrt(2)

            distances = [[0,     L,      L,    L,     L,    D],
            			[L,      0,      L,    0,     L,    L],
            	        [L,      L,      0,    L,     D,    L],     
            			[L,      0,      L,    0,     L,    L],     
            			[L,      L,      D,    L,     0,    L],     
            			[D,      L,      L,    L,     L,    0]]
            

        if formation_shape == 'cube':
            NN = 8
            L = 2
            d = np.sqrt(2)*L
            D = np.sqrt(3)*L
            distances = [[0,    L,      d,    L,    D,	 0,	  L,  	d],
                        [L,     0,      L,    0,    d, 	 D,	  0, 	L],
                        [d,     L,      0,    L,    L,   d,   D,    0],     
                        [L,     0,      L,    0,    0,   L,   d,    D],         
                        [D,     d,      L,    0,    0,   L,   d,    L], 
                        [0,     D,      d,    L,    L,	 0,	  L,  	0],
                        [L,     0,      D,    d,    d, 	 L,	  0, 	L],
                        [d,     L,      0,    D,    L,   0,   L,    0]]
            
        

        self.dist = np.asarray(distances)
        Adj = self.dist > 0
        self.neigh = np.where(Adj[:, self.agent_id]>0)[0]
        self.neigh = list(map(int, self.neigh))


        # create a subscription to each neighbor
        for j in self.neigh:
            self.create_subscription(
                                    MsgFloat, 
                                    f'/topic_{j}', #  topic_name
                                    self.listener_callback, 
                                    10)
        
        # create the publisher
        self.publisher = self.create_publisher(
                                            MsgFloat, 
                                            f'/topic_{self.agent_id}',
                                            10)
        timer_period = 0.5 # [seconds]
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # definite initial positions and add to the dictionary
        p_init = np.random.rand(self.n_p*NN, 1)
        p_init = list(map(float, p_init))

        if formation_shape == 'pentagon':
            for i in range(2, self.n_p*NN, 3):
                p_init[i] = 0.0

        # initialize a dictionary with a list of received messages from each neighbor j [a queue]
        self.pp_k = { j: [p_init[j*self.n_p : j*self.n_p+self.n_p]] for j in self.neigh }

        self.pp_k[self.agent_id] = [p_init[self.agent_id*self.n_p : self.agent_id*self.n_p+self.n_p]]

        print(f"Setup of agent {self.agent_id} completed")

    def listener_callback(self, msg):
        self.pp_k[int(msg.data[0])].append(list(msg.data[1:]))
    
        # syncronization check
        if all( len(value) != 0 for value in self.pp_k.values()) :
            
            self.all_received = True
            self.get_logger().info("A MESSAGE ARRIVED FROM ALL NEIGHBOURS")


    def timer_callback(self):

        if self.all_received:
            self.get_logger().info(f"ITERATION {self.kk}")
            self.get_logger().info(f"AGENTS POSITION: {self.pp_k}")
            pp_ii = self.pp_k[self.agent_id].pop(0)
            pp_ii = np.asarray(pp_ii)
            p_ii_kk = pp_ii

            for jj in self.neigh:       
                pp_jj = self.pp_k[jj].pop(0)
                pp_jj = np.asarray(pp_jj)

                dV_ij = (np.linalg.norm(pp_ii-pp_jj)**2 - self.dist[self.agent_id,jj]**2)*(pp_ii - pp_jj)		#gradient of Vij

                p_ii_kk = p_ii_kk - self.dt*dV_ij

            
            # Stop the node if kk exceeds the maximum iteration
            if self.kk > self.max_iters:
                print("\nMAXITERS reached")
                sleep(3) # [seconds]
                self.destroy_node()

            # Publish the updated message
            msg = MsgFloat()
            msg.data = [ float(self.agent_id)] + [float(i) for i in p_ii_kk]
            self.pp_k[self.agent_id] = [list(map(float, p_ii_kk))]
            self.publisher.publish(msg)

            self.get_logger().info(f"THE AGENT {self.agent_id} UDATED ITS STATUS AS: {p_ii_kk}")
            
            # update iteration counter and flag for syncronization
            self.kk += 1
            self.all_received = False

def main():
    rclpy.init()

    agent = Agent()
    agent.get_logger().info(f"Agent {agent.agent_id:d} -- Waiting for sync...")
    sleep(1)
    agent.get_logger().info("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()