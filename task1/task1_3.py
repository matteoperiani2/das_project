import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

from src.data_preparation import *
from src.modelling_nn import *
from src.utils import *

###############################################################################
# Distrbuted setting
NN = 5

# Dataset Settings
TARGET = 7
TRAIN_AGENT_DATA = 256
TEST_AGENT_DATA = 64
TRAIN_SIZE = TRAIN_AGENT_DATA * NN
TEST_SIZE = TEST_AGENT_DATA * NN

# Training parameters
N_EPOCHS = 1000
STEP_SIZE = 0.005

# saving 


###############################################################################
fix_seed(42)

## DATASET PREPARATION
x_train, y_train, x_test, y_test = prepare_dataset("task1/dataset/",
                                                    TARGET,
                                                    TRAIN_SIZE,
                                                    TEST_SIZE,
                                                    (int(np.sqrt(d)), int(np.sqrt(d))))

x_train, y_train = split_images_per_agents(x_train, y_train, NN, TRAIN_AGENT_DATA)
x_test, y_test = split_images_per_agents(x_test, y_test, NN, TEST_AGENT_DATA)
print(f"\nTrain data size: {x_train.shape}")
print(f"\nTrain laebl size: {y_train.shape}")
print(f"\nTest data size: {x_test.shape}")
print(f"\nTest laebl size: {y_test.shape}")

for i in range(NN):
    print(Counter(y_train[i]))
# exit()

print(f"\nTrain size: {x_train.shape}")

#  Generate Network Graph
G = generate_graph(NN, graph_type="Cycle")
# G = generate_graph(N_AGENTS, graph_type="Path")
# G = generate_graph(N_AGENTS, graph_type="Star")

ID_AGENTS = np.identity(NN, dtype=int)

while 1:
	ADJ = nx.adjacency_matrix(G)
	ADJ = ADJ.toarray()	

	test = np.linalg.matrix_power((ID_AGENTS+ADJ),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()

# METROPOLIS HASTING
WW = metropolis_hasting(ADJ, NN)

print('Row Stochasticity {}'.format(np.sum(WW,axis=1)))
print('Col Stochasticity {}'.format(np.sum(WW,axis=0)))

# Network Variables
xx = np.zeros((T, d))
weights = np.random.randn(T-1, d, d+1)*1e-1
uu = np.array([weights.copy() for _ in range(NN)])
uu_kp1 = np.zeros((NN, T-1, d, d+1))
ss = np.zeros((NN, T-1, d, d+1))
ss_kp1 = np.zeros((NN, T-1, d, d+1))
grads = np.zeros((NN, T-1, d, d+1))

# plot variables
consensus_error = np.zeros((N_EPOCHS, NN))
J = np.zeros((N_EPOCHS, NN)) # Cost function
NormGradJ = np.zeros((N_EPOCHS, NN))

# Init of Gradient Tracking Algorithm
for agent in range(NN):
    agent_grad = 0
    for img in range(TRAIN_AGENT_DATA):            
        xx = forward_pass(uu[agent], x_train[agent, 0])

        loss_grad = np.zeros((d))
        loss, loss_grad[0] = binary_cross_entropy(xx[-1, 0], y_train[agent, 0])

        _, grad = backward_pass(xx, uu[agent], loss_grad)
        agent_grad += grad / TRAIN_AGENT_DATA
        
    grads[agent] = agent_grad
    ss[agent] =  agent_grad

print()
###############################################################################
# TRAIN
for epoch in range(N_EPOCHS):
    for agent in range(NN):

        neighs = np.nonzero(ADJ[agent])[0]

        # Gradient Tracking Algorithm - Weights Update
        uu_kp1[agent] = (WW[agent, agent] * uu[agent]) - (STEP_SIZE * ss[agent])
        for neigh in neighs:
            uu_kp1[agent] += WW[agent, neigh] * uu[neigh]

        agent_grads = 0
        for img in range(TRAIN_AGENT_DATA):
            # Forward pass
            xx = forward_pass( uu_kp1[agent], x_train[agent, img])

            # Loss evalutation
            loss_grad = np.zeros((d))
            loss, loss_grad[0] = binary_cross_entropy(xx[-1, 0], y_train[agent, img])

            # Backward pass
            _, grad = backward_pass(xx, uu_kp1[agent], loss_grad)
            agent_grads += grad / TRAIN_AGENT_DATA

            J[epoch, agent] += loss / TRAIN_AGENT_DATA

            NormGradJ[epoch, agent] += np.linalg.norm(grad) / TRAIN_AGENT_DATA

        # Gradient Tracking Algorithm - SS Update
        ss_kp1[agent] = (WW[agent, agent] * ss[agent]) + (agent_grads - grads[agent])
        for neigh in neighs:
            ss_kp1[agent] += WW[agent, neigh] * ss[neigh]

        grads[agent] = agent_grads

    for agent in range(NN): 
        uu[agent] = uu_kp1[agent]
        ss[agent] = ss_kp1[agent]

    uu_mean = np.mean(uu, axis=0)
    for agent in range(NN):
        consensus_error[epoch, agent] = np.linalg.norm(uu_mean - uu[agent])
    
    if epoch % 1 == 0:
        print(f'Iteration n° {epoch:d}: loss = {np.mean(J[epoch]):.4f}, grad_loss = {np.mean(NormGradJ[epoch]):.4f}')

print(f'Iteration n° {epoch:d}: loss = {np.mean(J[epoch]):.4f}, grad_loss = {np.mean(NormGradJ[epoch]):.4f}') # last iteration

# Computes the mean error over uu
print("\nAgent erros:")
uu_mean = np.mean(uu, axis=0)
for agent in range(NN):
    print(f' - Agent {agent} mean_error = {np.linalg.norm(uu_mean - uu[agent]):.6f}')

###############################################################################
# TEST
if 1:
    agent = 0
    good = 0
    for img in range(TRAIN_AGENT_DATA):
        # in this way, as we did for computation of BCE, we got only the output value of the network
        pred = round(forward_pass(uu[agent], x_train[agent, img])[-1, 0])
        if pred == y_train[agent, img]:
            good +=1
                
    print(f"\nAccuracy of agent 0 on train set: {good/TRAIN_AGENT_DATA*100:.2f} %")

if 1:
    agent = 0
    good = 0
    for img in range(TEST_AGENT_DATA):
        # in this way, as we did for computation of BCE, we got only the output value of the network
        pred = round(forward_pass(uu[agent], x_test[agent, img])[-1, 0])
        if pred == y_test[agent, img]:
            good +=1
                        
    print(f"Accuracy of agent 0 on test set: {good/TEST_AGENT_DATA*100:.2f} %")

if 0:
    agent_accuracy = np.zeros(shape=NN)
    good_classification = np.zeros(shape=NN)
    for agent in range(NN):
        for img in range(TEST_AGENT_DATA):
            output = forward_pass(uu[agent], x_test[agent, img])
            pred = 1 if output[-1, 0] >= 0.5 else 0
            if pred == y_test[agent, img]:
                good_classification[agent] +=1
        agent_accuracy[agent] = good_classification[agent] / TEST_AGENT_DATA

    for agent in range(NN):
        print('\nAGENT: ', agent)
        print("Correctly classified point: ", good_classification[agent])
        print("Wrong classified point: ", (TEST_AGENT_DATA-good_classification[agent]))
        print(f"Accuracy: {agent_accuracy[agent]*100:.2f} %") 

    print(f"\nGlobal network accuracy: {np.mean(agent_accuracy)*100:.2f} %")

img_dir = os.path.join("task1/imgs/task13/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(img_dir)
###############################################################################
# PLOT

plt.figure('Cost function', figsize=(12,8))
plt.title('Evolution of the cost function')
plt.semilogy(range(N_EPOCHS),np.sum(J, axis=1)/NN, label='Total Normalized Cost Evolution', linewidth = 2)
for agent in range(NN):
     plt.semilogy(range(N_EPOCHS), J[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.ylabel(r'$\sum_{n=1}^N J$')
plt.legend()
plt.savefig(f"{img_dir}/cost_funct.png", )

plt.figure('Norm of Gradient function', figsize=(12,8))
plt.title('Evolution of the norm of the gradient of the cost function')
plt.semilogy(range(N_EPOCHS), np.sum(NormGradJ, axis=-1)/NN, label='Total Norm Gradient Evolution', linewidth = 2)
for agent in range(NN):
    plt.semilogy(range(N_EPOCHS), NormGradJ[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.ylabel(r"$|| \nabla J_w(x_t^k) ||_2$")
plt.legend()
plt.savefig(f"{img_dir}/norm_grad_cost.png")

plt.figure('Evolution of the  error', figsize=(12,8))
plt.title('Evolution of agent weights error from the mean value')
for agent in range(NN):
    plt.semilogy(range(N_EPOCHS), consensus_error[:, agent], label=f"agent{agent+1}")
plt.xlabel('Updates')
plt.ylabel(r'$|| u^\star - u_i ||_2, \quad u^\star = \frac{1}{N} \sum_{i=1}^N u_i \quad \forall i=1,\dots,N $')
plt.legend(loc="upper right")
plt.savefig(f"{img_dir}/consensus_erros.png")

# plt.show()