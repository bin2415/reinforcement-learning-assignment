# !/usr/bin/env python3
'''
@author: binpang
@time: 2017/12/10
'''
import gym
import tensorflow as tf
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, session, num_inputs, num_outputs,learning_rate = 0.001):
        '''
        MLP Network's initial function

        Args:
            session: tensorflow's session
            num_inputs: input number
            num_outputs: output number
        '''
        self.session = session
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.data_input = tf.placeholder(tf.float32, [None, num_inputs], name = 'input')
        self.action = tf.placeholder(tf.int32, [None], name = 'action')
        self.target = tf.placeholder(tf.float32, [None], name = 'target')
        action_onehot = tf.one_hot(self.action, self.num_outputs, name = 'actiononehot')


        fc1 = self.fc_layer(self.data_input, shape=(num_inputs, 8), name = 'dqn_fc1')
        fc2 = self.fc_layer(fc1, shape = (8, 16), name = 'dqn_fc2')
        #fc3 = self.fc_layer(fc2, shape = (32, 16), name = 'dqn_fc3')
        #fc4 = self.fc_layer(fc3, shape = (32, 8), name = 'dqn_fc4')
        self.data_output = self.fc_layer(fc2, shape = (16, num_outputs), name = 'dqn_output', relu = False)
        #self.data_output = fc4
        #fc1 = tf.layers.dense(self.data_input, 8, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1), name = "dqn_fc1")
        #fc2 = tf.layers.dense(fc1, 16, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1), name = "dqn_fc2")
        #fc3 = tf.layers.dense(fc2, 32, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1), name = "dqn_fc3")
        #fc4 = tf.layers.dense(fc3, 8, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1), name = "dqn_fc4")
        #self.data_output = tf.layers.dense(fc4, self.num_outputs, kernel_initializer=tf.random_normal_initializer(0, 0.1), name = "dqn_dataoutput")

        pward = tf.reduce_sum(tf.multiply(self.data_output, action_onehot), reduction_indices = 1)
        #varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqn_")
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, pward))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
    
    def update(self, state, action, reward):
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict= {self.data_input: state, self.action:action, self.target:reward})
        return loss

    def saveModel(self, model_name):
        self.saver.save(self.session, model_name)
    
    def restoreModel(self, model_name):
        self.saver.restore(self.session, model_name)

    def predict(self, state):
        '''
        Ret:
            predict values 
        '''
        return self.session.run(self.data_output, feed_dict={self.data_input:state})
    
    def fc_layer(self, x, shape, name, relu = True):
        num_inputs, num_outputs = shape
        W = self.weight_variable(shape, 1.0, name + '/W')
        b = self.bias_variable([num_outputs], 0.0, name + '/b')
        if relu:
            return tf.nn.relu(tf.add(tf.matmul(x, W), b))
        else:
            return tf.add(tf.matmul(x, W), b)
    
    def weight_variable(self, shape, std, name):
        initial = tf.truncated_normal(shape, stddev = std)
        W = tf.Variable(initial, name = name)
        return W
    
    def bias_variable(self, shape, value, name):
        initial = tf.constant(value, shape = shape)
        b = tf.Variable(initial, name = name)
        return b
        

class DQN:
    def __init__(self, env, session, episode = 10000, max_t = 200, learning_rate = 0.0005, explore_rate = 0.01, discount_factor = 0.95, max_memory = 20000, reward_func = None, model_name = None):
        '''
            DQN initial function 
            Args:
                env: gym environment
                session: tensorflow sesssion
        '''
        self.environment = env
        self.session = session
        self.episode = episode
        self.max_t = max_t
        self.learning_rate = learning_rate
        self.explore_rate = explore_rate
        self.discount_factor = discount_factor
        self.max_memory = max_memory
        self.reward_func = reward_func
        self.model_name = model_name

        num_inputs = int(env.observation_space.shape[0])
        num_outputs = int(env.action_space.shape[0])
        self.mlp = MLP(session, num_inputs, num_outputs)
        self.memory = list()
        
    
    def learning(self, batch_size):
        '''
            Learning process
        '''
        explore_rate = self.get_explore_rate(0)
        step_index =0
        loss_list = list()
        reward_list = list()
        for epi in range(self.episode):
            total_reward = 0
            state_old = self.environment.reset()
            temp_loss_list = list()
            for t in range(self.max_t):
                if random.random() < explore_rate:
                    action = self.environment.action_space.sample()
                else:
                    temp = self.mlp.predict(state[np.newaxis, :])
                    action = np.argmax(temp[0])
                
                state, reward, done, _ = self.environment.step(action)
                total_reward += reward
                #if done:
                    #reward = 200 + total_reward
                #else:
                    #reward = abs(state[0] - state_old[0]) * 10
                    #if reward < 0:
                    #    reward = 0
                
                if self.reward_func != None:
                    reward = self.reward_func(self.environment, state, state_old, done)

                self.memory.append((state_old, action, reward, state, done))
                state_old = state
                if len(self.memory) > batch_size:
                    sample_data = random.sample(self.memory, batch_size)
                    totrain = list()
                    next_states = [data[3] for data in sample_data]
                    pred_reward = self.mlp.predict(next_states)
                    #print(pred_reward)

                    for b_index in range(batch_size):
                        temp_state, temp_action, temp_reward, temp_next_state, temp_done = sample_data[b_index]
                        predict_action = max(pred_reward[b_index])
                        #print(predict_action)
                        

                        if temp_done:
                            yj = temp_reward
                        else:
                            yj = temp_reward + self.discount_factor * predict_action

                        totrain.append([temp_state, temp_action, yj])
                    
                    ##update
                    states = [k[0] for k in totrain]
                    actions = [k[1] for k in totrain]
                    rewards = [k[2] for k in totrain]

                    loss = self.mlp.update(states, actions, rewards)
                    temp_loss_list.append(loss)

                    if len(self.memory) > self.max_memory:
                        self.memory = self.memory[1:]
               
                if done or t >= self.max_t-1:
                    reward_list.append(total_reward)
                    if len(temp_loss_list) > 0:
                        loss_list.append(sum(temp_loss_list)/len(temp_loss_list))

                    print("Episode %d finished %d" % (epi, t))
                    break

            explore_rate = self.get_explore_rate(epi)

        if  self.model_name != None:  
            self.mlp.saveModel(self.model_name)
        
        #plot the figure
        fig = plt.figure()
        plt.plot(range(len(loss_list)), loss_list)
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.savefig("loss_dqn.png")
        plt.close('all')

        fig = plt.figure()
        plt.plot(range(self.episode), reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.savefig("reward_dqn.png")
        plt.close('all')

                    
    def test(self, test_num, max_trace):

        if self.model_name != None:
            self.mlp.restoreModel(self.model_name)
        #self.environment = gym.wrappers.Monitor(self.environment, './DQNResult/CartPoleVideo.mp4', force=True)
        reward_list = list()

        for epi in range(test_num):
            temp_reward = 0
            state = self.environment.reset()
            for t in range(max_trace):
                #self.environment.render()
                temp = self.mlp.predict(state[np.newaxis, :])
                action = np.argmax(temp[0])
                state, reward, done, _ = self.environment.step(action)
                temp_reward += reward
                if done or t >= max_trace - 1:
                    print("Episode %d, step %d" %(epi, t))
                    reward_list.append(temp_reward)
                    break
        
        plt.plot(range(len(reward_list)), reward_list)
        plt.xlabel("Episode")
        plt.ylabel("reward")
        plt.savefig("reward_qlearning.png")
        plt.close('all')

        print("Testing %d episode" % epi)
        print("The testing average %f" % (sum(reward_list) / len(reward_list)))
        print("The testing std %f" %(np.std(reward_list, ddof = 1)))
        # reward list, can calculate the average number and standard deviation
        return reward_list


    
    def get_explore_rate(self, t):
        return max(self.explore_rate, min(1, 1.0 - math.log10((t+1)/10)))
            
