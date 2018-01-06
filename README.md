## What has done
- Using Q-learning to train gym cartpole-v0, mountainCar-v0, acrobot-v1
- Using DQN to train gym cartpole-v0, mountainCar-v0, acrobot-v1
- Using Double-DQN to train gym cartpole-v0, mountainCar-v0, acrobot-v1

## Environment 
- python3.5
- tensorflow
- numpy
- gym

## How to run
- If you want to run Q-learning, please run ```python testQLearning.py -c[-m -a]``` where -c will run the cartpole-v0, -m will run the mountainCar-v0, -a will run the acrobot-v1. The cartpole-v0 average is about 4000 steps, mountaincar is about 150, the acrobot is about 88.
- If you want to run DQN, please run ```python testDQN.py -c[-m -a]```. The cartpole-v0 is very well, which will last more than 20000 steps, the mountaincar is about 145, the acrobot is about 115.
- If you want to run Double-DQN, please run ```python testImprovedDQN.py -c[-m -a]```. The cartpole is very well, which will last more than 20000 steps, the mountaincar is about 147, the acrobot is about 98.

## Result
- The videos folder saves the cartpole render videos.