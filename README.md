# Active, Semi-supervised Reinforcement Learning
Blog
# Setup
The libriary requires:<br>
  * gym <br>
  * Tensorflow (& keras) <br>
  * numpy <br>
  * matplotlib <br>
  * math <br>
  * random <br>
  * argparse <br>
  
# CartPole
>Cartpole is a classic control game from OpenAI Gym, and the goal of the game is to prevent the pendulum from both falling over and >moving away from center. The action player can take is to push the cart to either left or right with a fixed force, and the state the >cartpole has is the position and acceleration of both cart and pole. The game is considered terminal once the pole is more than 12 >degrees from vertical, or once the cart moves more than 2.4 units from the center. [Source](https://gym.openai.com/envs/CartPole-v1/)


# Agents
## About
There are in total 4 agents:
* Agent_f <br>
* Agent_a <br>
* Agent_e <br>
* Agent_r <br>
For a particular ratio `r` and total timesteps `t`, Agent--e obtains reward in the first `t*r` timesteps, Agent--r randomly gets reward with probability `r` at each step, and Agent--a decides when to ask for reward, so all agents have same amount of training set. We compare the performance of each agent when they receive reward over different ratios.
## DQN
Standard DQN with experience replay
### Hyperparameters:
* GAMMA = 0.99
* LEARNING_RATE = 0.0005
* MEMORY_SIZE = 10000
* BATCH_SIZE = 32
* EXPLORATION_MAX = 1.0
* EXPLORATION_MIN = 0.1
### Model Structure
1. Dense layer - input: 4, output: 32, activation: relu <br>
2. Dense layer - input 32, output: 32, activation: relu <br>
3. Dense layer - input 32, output: 2 <br>
* logcosh loss function
* Adam optimizer

## RNN
Reward neueal network
### Hyperparameters:
* LEARNING_RATE = 0.001
* BATCH_SIZE = 32
### Model Structure
#### Classifier DNN
Predict whether the next state is terminal
1. Dense layer - input: 4, output: 32, activation: relu, kernel_initializer: normal <br>
2. Dense layer - input 32, output: 1, activation: sigmoid <br>
* binary_crossentropyloss function
* Adam optimizer
#### Regression DNN
Predict the value of reward function
1. Dense layer - input: 4, output: 64, activation: relu, kernel_regularizer: regularizers.l1(0.001), kernel_initializer: normal <br>
2. Dense layer - input 64, output: 256, activation: relu, kernel_regularizer: regularizers.l1(0.001), kernel_initializer: normal <br>
3. Dense layer - input 256, output: 1 <br>
* mae loss function
* Adam optimizer

# Performance
example of score history
![](https://github.com/RoujiaD/semi-supervisedRL/blob/master/scores/history.png)

test scores vs. ratio
![](https://github.com/RoujiaD/semi-supervisedRL/blob/master/scores/test_score.png)

Contents and Usage
=======================
`Agent_a.py`: This script trains a DQN which asks for reward when it's uncertain about its predictions, and in the end it will save the trained DQN (so you need to create a directory before running the script) and output the predicted scores and actual scores. <br>
`Agent_f.py`: This scipt trains a DQN which gets reward at every step, and in the end it will save the trained DQN (so you need to create a directory before running the script) and output score history. <br>
`Agent_e.py`: This script trains a DQN which gets reward in the beginning, and in the end it will save the trained DQN (so you need to create a directory before running the script) and output the predicted scores and actual score. <br>
`Agent_r.py`: This script trains a DQN which randomly receives reward at each step, and in the end it will save the trained DQN (so you need to create a directory before running the script) and output the predicted scores and actual score. <br>
`RNN.py`: This script stores functions related to the RNN <br>
`test agents`: This is used to test the performance of all agents by loading the trained DQNs and test the mean score they obtain in 50 trials.

