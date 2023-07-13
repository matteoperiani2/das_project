import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_preparation import prepare_dataset
from src.modelling_nn_new import *
from src.utils import *

if len(sys.argv) != 6 and (len(sys.argv) != 2 or sys.argv[1] != "default"):
 print("Bad usage, launch the scrit with one of the following line:")
 print("python3 task1_2.py <TARGET> <MAX_EPOCHS> <TRAIN_SIZE> <TEST_SIZE> <STEP_SIZE>")
 print("or")
 print("python3 task1_2.py default")
 exit()
elif sys.argv[1] == "default":
    TARGET = 7
    TRAIN_SIZE = 512
    TEST_SIZE = 256
    MAX_EPOCHS = 20
    STEP_SIZE = 1e-2
else:
    TARGET = int(sys.argv[1])
    MAX_EPOCHS = int(sys.argv[2])
    TRAIN_SIZE = int(sys.argv[3])
    TEST_SIZE = int(sys.argv[4])
    STEP_SIZE = float(sys.argv[5])

BATCH_SIZE = 32
N_BATCH = int(TRAIN_SIZE / BATCH_SIZE)

print("Paramters configuration:")
print(f" - TARGET={TARGET}")
print(f" - TRAIN_SIZE={TRAIN_SIZE}")
print(f" - TEST_SIZE={TEST_SIZE}")
print(f" - MAX_EPOCHS={MAX_EPOCHS}")
print(f" - STEP_SIZE={STEP_SIZE}")
print()

fix_seed(42)

## DATASET PREPARATION
x_train, y_train, x_test, y_test = prepare_dataset("dataset/",
                                                    TARGET,
                                                    TRAIN_SIZE,
                                                    TEST_SIZE,
                                                    (int(np.sqrt(input_shape)), int(np.sqrt(input_shape))))

print("Preprocessing complete!")
print()
print("Start training...")

## INITIALIZATION
uu = [1e-1*np.random.randn(layer_neurons[l_idx+1], layer_neurons[l_idx]+1) for l_idx in range(T-1)]

J = np.zeros(MAX_EPOCHS)
NormGradJ = np.zeros(MAX_EPOCHS)

print()

# TRAIN
for epoch in range(MAX_EPOCHS):
    for batch_iter in range(N_BATCH):

        batch_grads = [np.zeros_like(u_l) for u_l in uu]
        for i in range(BATCH_SIZE):
            img = batch_iter*BATCH_SIZE + i
            
            xx = forward_pass(uu, x_train[img]) 

            # loss ( using only the output of the first neuron ) & mask output gradient
            loss, grad_loss = binary_cross_entropy(xx[-1][0], y_train[img])

            # backward pass
            _, grads = backward_pass(xx, uu, grad_loss)
            for l in reversed(range(T-1)):
                batch_grads[l] += grads[l] / BATCH_SIZE

            # loss loss grad accumulation
            J[epoch] += loss / TRAIN_SIZE
            for l in range(T-1):
                NormGradJ[epoch] += (np.abs(batch_grads[l]).sum() / batch_grads[l].size / N_BATCH)

		# gradient algorithm
        for l in range(T-1):
            uu[l] = uu[l] - STEP_SIZE*batch_grads[l]
    
    if epoch % 1 == 0:
        print(f'Iteration n° {epoch+1:d}: loss = {J[epoch]:.4f}, grad_loss = {NormGradJ[epoch]:.4f}')

print()
print("Training complete!")
print()
print("Evaluation of the network:")

### TEST
if 1:
    good_match = 0
    for img in range(TRAIN_SIZE):       
        pred = round(forward_pass(uu, x_train[img])[-1][0])
        if pred == y_train[img]:
            good_match +=1
        
    print(f" - Accuracy on train set: {good_match/TRAIN_SIZE*100:.2f} %")

if 1:
    good_match = 0
    for img in range(TEST_SIZE):       
        pred = round(forward_pass(uu, x_test[img])[-1][0])
        if pred == y_test[img]:
            good_match +=1
        
    print(f" - Accuracy on test set: {good_match/TEST_SIZE*100:.2f} %")

img_dir = os.path.join("imgs/task12/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(img_dir)

f = open(f"{img_dir}/parms.txt", "a")
f.write(f"TARGET = {TARGET}\n")
f.write(f"TRAIN_SIZE = {TRAIN_SIZE}\n")
f.write(f"TEST_SIZE = {TEST_SIZE}\n")
f.write(f"MAX_EPOCHS = {MAX_EPOCHS}\n")
f.write(f"STEP_SIZE = {STEP_SIZE}\n")
f.close()

## PLOT
plt.figure(figsize=(12, 8))
plt.title('Cost Function Evolution')
plt.semilogy(J, label='Cost Evolution', linewidth = 3)
plt.xlabel("Epochs")
plt.ylabel("J")
plt.savefig(f"{img_dir}/loss.png")

plt.figure(figsize=(12, 8))
plt.title('Norm of the Cost Function Gradient Evolution')
plt.semilogy(NormGradJ, label='Total Gradient Evolution', linewidth = 3)
plt.xlabel("Epochs")
plt.ylabel(r"$||\Delta u_t||_2$")
plt.savefig(f"{img_dir}/grad_loss.png")
