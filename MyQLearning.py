# !/usr/bin/env python3
'''
@author: binpang
@time: 2017/12/09
'''

import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, num_space, bounds, episode=1000, max_t=200, learning_rate = 0.1, explore_rate = 0.01, discount_factor = 0.95, reward_func = None):
        '''
            QLearing init function

            Args:
                env: enviroument
                num_space: the discretized state's space
                bounds : environment bounds
                episode: training episode
                max_t: max track
                learning_rate: learning rate
                explore_rate: explore rate
                discount_facotr: discount factor
        '''
        self.envionment = env
        self.bounds = bounds
        self.learning_rate = learning_rate
        self.explore_rate = explore_rate
        self.qtable = np.zeros(num_space + (self.envionment.action_space.n,))
        self.num_space = num_space
        self.discount_factor = discount_factor
        self.episode = episode
        self.max_track = max_t
        self.reward_func = reward_func

    def learning(self):
        '''
            Q-learning learing
        '''
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        scores = list()

        for epi in range(self.episode):
            score = 0
            obv_old = self.envionment.reset()
            state_old = self.discretization(obv_old)

            for t in range(self.max_track):
                #self.envionment.render()

                #explore the action
                if random.random() < explore_rate:
                    action = self.envionment.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state_old])
                
                obv, reward, done, _ = self.envionment.step(action)
                score += reward
                if self.reward_func != None:
                    reward = self.reward_func(self.envionment, obv, obv_old, done)
                    #print(reward)
                #print(reward)

                state = self.discretization(obv)

                best_reward = np.amax(self.qtable[state])

                ##update the qtable
                #print(action)
                #print(state_old +(action,))
                self.qtable[state_old + (action,)] = (1-learning_rate)* self.qtable[state_old + (action,)] + learning_rate * (reward + self.discount_factor * best_reward)
    
                state_old = state
                obv_old = obv

                if done or t >= self.max_track - 1:
                    scores.append(score)
                    print("Episode %d finished after track %d, score is %d, mean score is %f" % (epi, t, score, np.mean(scores))) 
                    break
            explore_rate = self.get_explore_rate(epi)
            learning_rate = self.get_explore_rate(epi)           

    def test(self, test_time, max_trace):
        #self.envionment = gym.wrappers.Monitor(self.envionment, './QLearningResult/CartPoleVideo.mp4', force=True)
        reward_list = list()
        for epi in range(test_time):
            temp_reward = 0
            obv = self.envionment.reset()
            state = self.discretization(obv)
            for t in range(max_trace):
                #self.envionment.render()
                action = np.argmax(self.qtable[state])
                obv, reward, done, _ = self.envionment.step(action)
                temp_reward += reward
                if done or t >= max_trace - 1:
                    print("Episode %d, step %d" %(epi, t))
                    reward_list.append(temp_reward)
                    break
                state = self.discretization(obv)
        
        plt.plot(range(len(reward_list)), reward_list)
        plt.xlabel("Episode")
        plt.ylabel("reward")
        plt.savefig("reward_qlearning.png")
        plt.close('all')

        print("Testing %d episode" % test_time)
        print("The testing average %f" % (sum(reward_list) / len(reward_list)))
        print("The testing std %f" %(np.std(reward_list, ddof = 1)))
        
        # reward list, can calculate the average number and standard deviation
        return reward_list

                

        
    def discretization(self, state):
        '''
        discretize the state

        Args:
            bounds: state's bounds
            num_space: the discretized state's space
            state: the state that discretized
        Rets:
            state that has been discretized
        '''
        ret_result = list()

        for i in range(len(state)):
            if state[i] <= self.bounds[i][0]:
                temp_result = 0
            elif state[i] >= self.bounds[i][1]:
                temp_result = self.num_space[i] - 1
            else:
                range_i = self.bounds[i][1] - self.bounds[i][0]
                offset = (self.num_space[i]-1) * self.bounds[i][0]/range_i
                scaling = (self.num_space[i]-1)/range_i
                temp_result = int(round(scaling*state[i] - offset))
            ret_result.append(temp_result)
        return tuple(ret_result)

    def get_explore_rate(self, t):
        return max(self.explore_rate, min(1, 1.0 - math.log10((t+1)/10)))
    
    def get_learning_rate(self, t):
        return max(self.learning_rate, min(0.5, 1.0 - math.log10((t+1)/10)))

