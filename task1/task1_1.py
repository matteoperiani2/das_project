import sys, os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy

from src.utils import *

if len(sys.argv) != 5 and (len(sys.argv) != 2 or sys.argv[1] != "default"):
	print("Bad usage, launch the scrit with one of the following line:")
	print("python3 task1_2.py <MAX_ITERS> <NN> <d> <graph_type>")
	print("or")
	print("python3 task1_2.py default")
	exit()
elif sys.argv[1] == "default":
	MAXITERS = 1000
	NN = 5 # number of agents
	d = 3 # dimension of the input
	graph_type="Cycle"
	# graph_type="Path"
	# graph_type="Star"
else:
 MAXITERS = int(sys.argv[1])
 NN = int(sys.argv[2])
 d = int(sys.argv[3])
 graph_type = sys.argv[4]

print("Paramters configuration:")
print(f" - MAXITERS={MAXITERS}")
print(f" - NN={NN}")
print(f" - d={d}")
print(f" - graph_type={graph_type}")
print()
fix_seed(42)

# Generate Network a Graph
G = generate_graph(NN, graph_type=graph_type)

# Generate weight matrix column and row stochastic
ID_AGENTS = np.identity(NN, dtype=int)

ADJ = 0
while 1:
	ADJ = nx.adjacency_matrix(G)
	ADJ = ADJ.toarray()	

	test = np.linalg.matrix_power((ID_AGENTS+ADJ), NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()

# METROPOLIS HASTING
WW = metropolis_hasting(ADJ, NN)

with np.printoptions(precision=4, suppress=True):
	print('Check Stochasticity:\n row: {} \n column {}'.format(
  	np.sum(WW,axis=1),
		np.sum(WW,axis=0)
	))

# Declare Cost Variables
Q = np.zeros((NN,d, d))
for ii in range(NN):
	T = scipy.linalg.orth(np.random.rand(d,d))
	D = np.diag(np.random.rand(d))*10
	Q[ii] = T.T@D@T
 
R = 10*(np.random.rand(NN, d)-1)

# Compute Optimal Solution
Q_centr = np.sum(Q,axis=0)
R_centr = np.sum(R,axis=0)

u_opt = -np.linalg.inv(Q_centr)@R_centr
j_opt = 0.5*u_opt@Q_centr@u_opt+R_centr@u_opt
print("u_opt =", u_opt)
print("J(u_opt) =", j_opt)

# Declare Algorithmic Variables
UU = np.zeros((NN,MAXITERS,d))
SS = np.zeros((NN,MAXITERS,d))

UU_init = 10*np.random.rand(NN, d)
UU[:,0] = UU_init

for ii in range (NN):
	_, SS[ii,0] = quadratic_fn(UU[ii,0],Q[ii],R[ii])

JJ = np.zeros((MAXITERS))

###############################################################################
# GO!
stepsize = 1e-2
print()
for kk in range (MAXITERS-1):

	if (kk % 100) == 0:
		print(f"Iteration: {kk:3d}")
	
	for ii in range (NN):
		Nii = np.nonzero(ADJ[ii])[0]

		UU[ii,kk+1] = WW[ii,ii]*UU[ii,kk] - stepsize*SS[ii,kk]
		for jj in Nii:
			UU[ii,kk+1] += WW[ii,jj]*UU[jj,kk]
		
		f_ii, grad_fii = quadratic_fn(UU[ii,kk],Q[ii],R[ii])
		_, grad_fii_p = quadratic_fn(UU[ii,kk+1],Q[ii],R[ii])

		SS[ii,kk+1] = WW[ii,ii]*SS[ii,kk] +(grad_fii_p - grad_fii)
		for jj in Nii:
			SS[ii,kk+1] += WW[ii,jj]*SS[jj,kk]
		
		JJ[kk] += f_ii

# Terminal iteration
for ii in range(NN):
	f_ii, grad_fii = quadratic_fn(UU[ii,-1],Q[ii],R[ii]) 
	JJ[-1] += f_ii # Last entry

print()
print(f"Estimate values of agent's u_opt at {kk+1} iteration:")
for i in range(NN):
	print(f"agent_{i+1}: {UU[i, -1]}")
print("Average of agent u_opt:", np.mean(UU[:, -1], axis=0))
print(f"Estimate values of J(u_opt) at {kk+1} iteration", JJ[-1])

# consensus error
UU_avg = np.mean(UU, axis=0) # shape (ITERS, dd)
EE = np.sum(
    np.linalg.norm(
        UU - UU_avg, 
        axis=2), # norm ||.||_R^d
    axis=0) # sum over i

print()
print(f"Consensus error at {kk+1} iteration:", EE[-1])

img_dir = os.path.join("imgs/task11/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(img_dir)

f = open(f"{img_dir}/parms.txt", "a")
f.write(f"MAXITERS = {MAXITERS}\n")
f.write(f"NN = {NN}\n")
f.write(f"d = {d}\n")
f.write(f"graph_type = {graph_type}\n")
f.close()

###############################################################################
# Figure 1 : Agent graph
plt.figure(figsize=(12, 8))
plt.title("Graph angent positions")
nx.draw(G, with_labels=True)
plt.savefig(f"{img_dir}/graph.png")

###############################################################################
# Figure 2 : Cost Evolution
plt.figure(figsize=(12, 8))
plt.title("Evolution of the cost")
plt.plot(np.arange(MAXITERS), np.repeat(j_opt,MAXITERS), '--', linewidth=3)
plt.plot(np.arange(MAXITERS), JJ)
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$\sum_{i=1}^N J_i(u_i^k)$, $f^\star$")
plt.savefig(f"{img_dir}/cost.png")

###############################################################################
# Figure 3 : Norm of the gradient of the cost evolution
plt.figure(figsize=(12, 8))
plt.title("Evolution of the norm of the function's gradient")
plt.semilogy(np.arange(MAXITERS), np.mean(np.linalg.norm(SS, axis=-1), axis=0), linewidth=2)
for i in range(NN):
	plt.semilogy(np.arange(MAXITERS), np.linalg.norm(SS[i], axis=-1), '--', label=f"agent_{i+1}", )
plt.legend()
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$|| s_i(u^k) ||_2, \frac{1}{N} \sum_{i=1}^N || s_i(u^k) ||_2$")
plt.savefig(f"{img_dir}/grad_loss.png")

###############################################################################
# Figure 4 : Consensus Error Evolution
plt.figure(figsize=(12, 8))
plt.title("Evolution of the consensus error")
plt.semilogy(np.arange(MAXITERS), EE)
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$\sum_{i=1}^N ||\bar{u}^k - u_i^k||_2$")
plt.savefig(f"{img_dir}/err.png")

