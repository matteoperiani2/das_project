<a name="br1"></a> 

Distributed Autonomous Systems

Course Project

The project consists in two main tasks. The ﬁrst one involves a data analytics application

while the second one deals with the control for multi-robot systems in ROS 2.

Task 1: Distributed Classiﬁcation via Neural Networks

Task 1 concerns the classiﬁcation of a set of images. Suppose to have a team of N agents.

Each agent i can access only to a private set containing m ∈ N images D

<sup>j</sup> ∈ R<sup>d</sup>

with

i

j

associated label y coming from a given dataset. The dataset is split in a training set

{D<sup>j</sup>, y<sup>j</sup>}

<sup>m</sup>train

and a test set {D , y }

h

h

<sup>m</sup>test

h=1

.

j=1

Two possible datasets are available:

(a) handwritten digits (mnist);

(b) Zalando’s article images (fashion mnist).

The assignment of datasets will depend on the group number. Odd groups will be assigned

option (a), while even groups will be assigned option (b). For instance, Group 23 must use

the mnist dataset, and Group 12 must use the fashion mnist dataset.

Task 1.1 – Distributed Optimization

1\. Implement the Gradient Tracking algorithm to solve a consensus optimization problem

in the form

X<sup>N</sup>

min

u

J<sub>i</sub>(u)

i=1

where J<sub>i</sub> is a quadratic function.

2\. Run a set of simulations to test the eﬀectiveness of the implementation. Moreover,

provide a set of solutions that includes diﬀerent weighted graph patterns (e.g., cycle,

path, star) whose weights are determined by the Metropolis-Hastings method. Finally,

for each simulation, plot the evolution of the cost function and of the norm of the

gradient of the cost function across the iterations.

Task 1.2 – Centralized Training

1\. Prepare the dataset for neural network training. Select a category (e.g., the label

“Sandal”)

(a) assign the label 1 to all the images belonging to the selected category;

(b) assign the label 0 to all other images, not belonging to the selected category;

(c) take only a reduced subset of images.

2\. Implement the multi-sample neural network, extending the one-sample example

presented during the lectures, with:

\- Sigmoid function as activation function σ(·);

1



<a name="br2"></a> 

\- Binary Cross-Entropy (BCE) as loss function `(·).

Note. Students can freely choose other activation and/or loss functions in their implemen-

tations. However, it is important to note that the evaluation will primarily focus on the

distributed implementation of the algorithm.

Task 1.3 – Distributed Training

1\. Split (randomly) the entire training set in N subsets, one for each agent i.

2\. Implement a distributed algorithm to train a neural network based on the Gradient

Tracking (you are allowed to extend the code provided during the lectures)

3\. Generate a set of simulations showing the convergence of the distributed algorithm

to a stationary point of the optimization problem. Moreover, plot the evolution of

the cost function and of the norm of the gradient of the cost function across the

iterations.

4\. Test diﬀerent dataset sizes (start with a small number of samples)

5\. Evaluate the quality of the obtained solution by computing its accuracy (say computed

by agent 0) on the test set. That is, compute the percentage of success of the following

test, for j = 1, . . . , m<sub>test</sub>

(

? >D<sup>j</sup> ≥

1

0

if (u )

0

j

?

j

yˆ = φ(u , D ) =

? >D<sup>j</sup>

if (u )

< 0.

j

The classiﬁer succeeds if yˆ = y .

j

Hints:

1\. Important: you are allowed to use the ﬁles provided during the exercise lectures

2\. Reshape and normalize the samples so that D<sup>j ∈</sup> [0, 1]<sup>784</sup>

3\. The dataset can be imported from the Keras Python Library (from keras.datasets

import mnist, fashion mnist)

2



<a name="br3"></a> 

Task 2: Formation Control

Consider a team of N robots. We denote the position of robot i ∈ {1, . . . , N} at time

t ≥ 0 with x (t) ∈ R , and with p

3N

3

<sup>k</sup> ∈ R<sup>3</sup>

its discretized version at time k

respectively the stack vector of the continuous

∈ N

. Hence, we

i

i

represent with x(t) ∈ R and p

<sup>k</sup> ∈ R<sup>3N</sup>

and the discrete positions.

Task 2.1 – Problem Set-up

1\. Implement in ROS 2 a discrete-time version of the Formation Control law based on

potential. Choose as potential function

ꢀ

ꢁ

1

4

2

2

2

ij

V<sub>ij</sub>(x) =

kx − x k − d

,

i

j

with d ∈ R the assigned distances between two robots. As a consequence, for each

ij

robot i:

ꢀ

ꢁ

X

2

2

x˙ (t) = f (x(t)) = −

kx (t) − x (t)k − d (x (t) − x (t))

i

i

i

j

ij

i

j

j∈N

i

k+1

i

k

i

k

p

= p + ∆ · f (p )

i

where ∆ > 0 is the sampling period (i.e., the node frequency in ROS 2). Start from

the Python ﬁles and the ROS 2 consensus package provided during the lectures.

2\. Run a set of simulations choosing diﬀerent formation patterns (e.g., letters, numbers,

polygons) and a diﬀerent number of agents, providing an animated visualization of

the team behavior (you can also use the RViz template provided during the exercise

lectures).

Task 2.2 – Collision Avoidance

1\. Implement a modiﬁed version of the Formation Control that includes collision

avoidance barrier functions. A candidate barrier function could be

2

V (x) = − log(kx (t) − x (t)k ).

ij

i

j

2\. Run a set of simulations showing the eﬀectiveness of the barrier functions.

Task 2.3 – Moving Formation and Leader(s) control

1\. Declare one agent (or more) as leader of the formation and implement a control law

(e.g., a proportional controller) to steer the formation toward a target position.

2\. Run a set of simulations choosing diﬀerent target positions.

Task 2.4 – (Optional) Obstacle Avoidance

1\. From Task 2.2, implement an obstacle avoidance algorithm;

2\. Run a set of simulations validating the algorithm.

3



<a name="br4"></a> 

Notes

1\. Each group must be composed of at most 3 students.

2\. Each group must attend at least 2 meetings with the tutor.

3\. All the emails for the project support must have the subject:

“[DAS2023]-Group X: support request”.

Moreover, all the group members, the tutor, and the professors must always be

included in the emails.

4\. The project report must be written in L<sup>A</sup>T X and must follow the main structure of

E

the provided template.

5\. The ﬁnal submission deadline is one week before the exam date.

6\. Final submission: one member of each group must send an email with subject

“[DAS2023]-Group X: Submission”, with attached a link to a OneDrive folder, con-

taining:

\- README.txt

\- report group XX.pdf

\- report – a folder containing the L<sup>A</sup>T X code and a figs folder (if any)

E

\- task 1 – a folder containing the code relative to Task 1, including README.txt

\- task 2 – a folder containing the code relative to Task 2, including README.txt

7\. Any other information and material necessary for the project development will be

given during project “meetings”.

4

