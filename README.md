# Duckievillage

Duckievillage is an adaptation of [Duckietown](https://duckietown.org)'s OpenAI Gym
[gym-duckietown](https://github.com/duckietown/gym-duckietown) for the undergraduate level [Intro
to Robotics](https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=MAC0318&nomdis=) (MAC0318)
course held at the [Institute of Mathematics and Statistics](http://www.ime.usp.br/) (IME) of the
[University of São Paulo](https://www5.usp.br/#english) (USP).

This project is not to be thought of as a fork or standalone project, but as a complement to
Duckietown's Gym. To be more precise, Duckievillage's objectives are two-fold: to be an environment
wrapper around `DuckietownEnv`, which allows a more custom-made environment for our own robotics
course; and to be used as course walkthrough.

---

## Installation

Duckievillage requires [Python](https://www.python.org/), [OpenAI Gym](https://gym.openai.com/),
[gym-duckietown](https://github.com/duckietown/gym-duckietown) and [PyTorch](https://pytorch.org/)
installed. If you're on Arch Linux, you can easily install everything except `gym-duckietown`
through the AUR:

```bash
yay -S python-gym-git python-pytorch
```

If you're on another distribution or OS, the second easiest way is through
[Anaconda](https://www.anaconda.com/). Simply download and run the installer. Once that's done,
install PyTorch and OpenAI Gym:

```bash
conda create -n duckievillage
conda activate duckievillage
conda install pip
conda install pytorch cudatoolkit=9.0 -c pytorch
pip3 install gym
```

To install `gym-duckietown`:

```bash
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
pip3 install -e .
```

## Usage

To import Duckievillage, simply copy `duckievillage.py` to your working directory and import it:

```python
import duckievillage
```

## Schedule

This is our current course schedule:

#### Lecture 1 - Introduction (07/08)

1. Intro to Robotics
2. Simulator (Duckievillage + `gym-duckietown`)

##### Classwork / Homework

1. Manual steering
2. Waypoint navigation

#### Lecture 2 - Route Planning 1 (14/08)

1. Route planning
2. Topological maps
3. Graph search

##### Classwork / Homework

1. BFS and A<sup>*</sup> on topological map to find and execute route
2. Execute route with finer route planning (e.g. smooth curves, drive only on right-side of lane,
   etc.)

#### Lecture 3 - Route Planning 2 (21/08)

1. Polygon maps
2. Potential fields

##### Classwork / Homework

1. "Online" potential field for solving the "catch" game (with duckies!)
2. Adapt potential field to usual traffic rules

#### Lecture 4 - Localization (28/08)

1. Bayesian filter
2. Histogram filter
3. Neural network classifier calibration

##### Classwork / Homework

1. 1D localization
2. 2D localization + calibration

#### BREAK (04/09)

#### Lecture 5 - Supervised Learning (11/09)

1. Supervised learning
2. Classification
3. Perceptrons, MLP
4. PyTorch

##### Classwork / Homework

1. Apply classification on the MNIST dataset

#### Lecture 6 - Imitation Learning (18/09)

1. Generalization
2. Visualization
3. Convolutional neural networks
4. Regression

##### Classwork / Homework

1. CNN filter visualization
2. Execute learned policy on the `gym-duckietown` simulator

#### Lecture 7 - Mobile Robots 1 (25/09)

1. Intro to mobile robots
2. Navigation
3. Lego NXT Mindstorm Interface
4. PID Control
5. Define final projects

##### Classwork / Homework

1. Robot assembly

#### Lecture 8 - Mobile Robots 2 (02/10)

1. Master-slave architecture
2. Raspberry Pi
3. Message passing
4. Navigation and sensoring via USB

##### Classwork / Homework

1. RC mobile robot
2. Image extraction via OpenCV

#### Work on final project (25/09 - 20/11)

#### Lecture 8 - Reinforcement Learning (27/11)

1. Reinforcement learning
2. Policy gradient
3. Deep Deterministic Policy Gradient (DDPG)

##### Classwork / Homework

1. Train and evaluate agent on the `gym-duckietown` simulator

#### Final Project Presentation (04/12)
