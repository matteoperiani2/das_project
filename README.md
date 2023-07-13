# Distributed Autonomous Systems

## Course Project

The project consists in two main tasks. The first one involves a data analytics application
while the second one deals with the control for multi-robot systems in ROS 2.

## Task 1: Distributed Classification via Neural Networks

Task 1 concerns the classification of a set of images. Suppose to have a team ofNagents.
Each agentican access only to a private set containingmi∈NimagesDj∈Rd with
associated labelyjcoming from a given dataset. The dataset is split in a training set
{Dj,yj}mj=1trainand a test set{Dh,yh}mh=1test.
Two possible datasets are available:

```
(a) handwritten digits (mnist);
```
```
(b) Zalando’s article images (fashionmnist).
```
The assignment of datasets will depend on the group number. Odd groups will be assigned
option (a), while even groups will be assigned option (b). For instance, Group 23 must use
themnistdataset, and Group 12 must use thefashionmnistdataset.

### Task 1.1 – Distributed Optimization

1. Implement theGradient Trackingalgorithm to solve a consensus optimization problem
    in the form

```
min
u
```
#### ∑N

```
i=
```
```
Ji(u)
```
```
whereJiis a quadratic function.
```
2. Run a set of simulations to test the effectiveness of the implementation. Moreover,
    provide a set of solutions that includes different weighted graph patterns (e.g., cycle,
    path, star) whose weights are determined by the Metropolis-Hastings method. Finally,
    for each simulation, plot the evolution of the cost function and of the norm of the
    gradient of the cost function across the iterations.

### Task 1.2 – Centralized Training

1. Prepare the dataset for neural network training. Select a category (e.g., the label
    “Sandal”)

```
(a) assign the label 1 to all the images belonging to the selected category;
(b) assign the label 0 to all other images, not belonging to the selected category;
(c) take only a reduced subset of images.
```
2. Implement the multi-sample neural network, extending the one-sample example
    presented during the lectures, with:
       - Sigmoid function as activation functionσ(·);


- Binary Cross-Entropy (BCE) as loss function`(·).

Note.Students can freely choose other activation and/or loss functions in their implemen-
tations. However, it is important to note that the evaluation will primarily focus on the
distributed implementation of the algorithm.

### Task 1.3 – Distributed Training

1. Split (randomly) the entire training set inNsubsets, one for each agenti.
2. Implement a distributed algorithm to train a neural network based on theGradient
    Tracking(you are allowed to extend the code provided during the lectures)
3. Generate a set of simulations showing the convergence of the distributed algorithm
    to a stationary point of the optimization problem. Moreover, plot the evolution of
    the cost function and of the norm of the gradient of the cost function across the
    iterations.
4. Test different dataset sizes (start with a small number of samples)
5. Evaluate the quality of the obtained solution by computing its accuracy (say computed
    by agent 0) on the test set. That is, compute the percentage of success of the following
    test, forj= 1,...,mtest

```
yˆj=φ(u?,Dj) =
```
#### {

```
1 if (u?)>Dj≥ 0
0 if (u?)>Dj< 0.
```
```
The classifier succeeds if ˆyj=yj.
```
Hints:

1. Important:you are allowed to use the files provided during the exercise lectures
2. Reshape and normalize the samples so thatDj∈[0,1]^784
3. The dataset can be imported from the Keras Python Library (from keras.datasets
    import mnist, fashionmnist)


## Task 2: Formation Control

Consider a team ofN robots. We denote the position of roboti∈ { 1 ,...,N}at time
t≥0 withxi(t)∈R^3 , and withpki∈R^3 its discretized version at timek∈N. Hence, we
represent withx(t)∈R^3 Nandpk∈R^3 Nrespectively the stack vector of the continuous
and the discrete positions.

### Task 2.1 – Problem Set-up

1. Implement in ROS 2 a discrete-time version of theFormation Controllaw based on
    potential. Choose as potential function

```
Vij(x) =
```
#### 1

#### 4

#### (

```
‖xi−xj‖^2 −d^2 ij
```
#### ) 2

#### ,

```
withdij∈Rthe assigned distances between two robots. As a consequence, for each
roboti:
```
```
x ̇i(t) =fi(x(t)) =−
```
#### ∑

```
j∈Ni
```
#### (

```
‖xi(t)−xj(t)‖^2 −d^2 ij
```
#### )

```
(xi(t)−xj(t))
```
```
pki+1=pki+ ∆·fi(pk)
```
```
where ∆>0 is the sampling period (i.e., the node frequency in ROS 2). Start from
the Python files and the ROS 2 consensus package provided during the lectures.
```
2. Run a set of simulations choosing different formation patterns (e.g., letters, numbers,
    polygons) and a different number of agents, providing an animated visualization of
    the team behavior (you can also use theRViztemplate provided during the exercise
    lectures).

### Task 2.2 – Collision Avoidance

1. Implement a modified version of theFormation Control that includes collision
    avoidance barrier functions. A candidate barrier function could be

```
Vij(x) =−log(‖xi(t)−xj(t)‖^2 ).
```
2. Run a set of simulations showing the effectiveness of the barrier functions.

### Task 2.3 – Moving Formation and Leader(s) control

1. Declare one agent (or more) as leader of the formation and implement a control law
    (e.g., a proportional controller) to steer the formation toward a target position.
2. Run a set of simulations choosing different target positions.

### Task 2.4 – (Optional) Obstacle Avoidance

1. From Task 2.2, implement an obstacle avoidance algorithm;
2. Run a set of simulations validating the algorithm.


### Notes

1. Each group must be composed of at most 3 students.
2. Each group must attend at least 2 meetings with the tutor.
3. All the emails for the project support must have the subject:
    “[DAS2023]-Group X:support request”.
       Moreover, all the group members, the tutor, and the professors must always be
       included in the emails.
4. The project report must be written in LATEX and must follow the main structure of
    the provided template.
5. The final submissiondeadlineisoneweek before the exam date.
6. Final submission: one member of each group must send an email with subject
    “[DAS2023]-Group X: Submission”, with attached a link to a OneDrive folder, con-
       taining:
          - README.txt
          - reportgroupXX.pdf
          - report– a folder containing the LATEX code and afigsfolder (if any)
          - task 1 – a folder containing the code relative to Task 1, includingREADME.txt
          - task 2 – a folder containing the code relative to Task 2, includingREADME.txt
7. Any other information and material necessary for the project development will be
    given during project “meetings”.


