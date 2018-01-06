from MyQLearning import *
from MyDQN import *
import tensorflow as tf
import gym
import getopt
import sys

def reward_func(env, state, state2, done):
    #if done:
    #    return -100

    x, x_dot, theta, theta_dot = state
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r2 + r1

def reward_func_car(env, state1, state2, done):
    if done:
        return 400
    else:
        return -1/abs(state1[0] - state2[0])

def reward_func_robot(env, state1, state2, done):
    if done:
        return 200
    return -1

def test_cartPole():
    env = gym.make('CartPole-v0')
    bounds = list(zip(env.observation_space.low, env.observation_space.high))
    zones = (1,2,20,10)
    bounds[1] = [-0.5, 0.5]
    bounds[3] = [-math.radians(50), math.radians(50)]
    env = env.unwrapped
    qlearning = QLearning(env, zones, bounds, reward_func = reward_func)
    qlearning.learning()
    qlearning.test(100, 20000)

def test_mountainCar():
    env = gym.make('MountainCar-v0')
    bounds = list(zip(env.observation_space.low, env.observation_space.high))
    zones = (20, 10)
    env = env.unwrapped
    qlearning = QLearning(env, zones, bounds, episode = 5000, reward_func = reward_func_car)
    qlearning.learning()
    qlearning.test(100, 2000)

def test_acrobot():
    env = gym.make('Acrobot-v1')
    bounds = list(zip(env.observation_space.low, env.observation_space.high))
    zones = (1, 1, 1, 1, 10, 10)
    env = env.unwrapped
    qlearning = QLearning(env, zones, bounds, episode = 1000, reward_func = reward_func_car)
    qlearning.learning()
    qlearning.test(100, 2000)

def usage():
    print("---------------Usage----------------")
    print("-h or --help, get the usage message")
    print("-c or --cartpole, run the cartpole-v0 gym envirmonent")
    print("-m or --mountaincar, run the mountaincar-v0 gym environment")
    print("-a or --acrobot, run the acrobot-v1 gym environment")
    print("--------------END Usage--------------")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)
    
    try:
        options, args = getopt.getopt(sys.argv[1:],"hcma",["help", "cartpole", "mountaincar", "acrobot"])
    except getopt.GetoptError:
        sys.exit(-2)
    
    for name, value in options:
        if name in ("-h", "-help"):
            usage()
        if name in ("-c", "--cartpole"):
            test_cartPole()
        if name in ("-m", "--mountaincar"):
            test_mountainCar()
        if name in ("-a", "--acrobot"):
            test_acrobot()
    
