#!/usr/bin/env python3

from gazebo_mantis_dqlearn import MantisGymEnv

import time
import os
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from collections import deque

import matplotlib.pyplot as plt
import sys
import signal

LIVE_PLOT = False  # Rise a new window to plot process while training

class Agent:
    '''
    Main class for agent
    '''
    def __init__(self, stateSize, actionSize):
        self.isTrainActive = True  # Train model (Make it False for just testing)
        self.loadModel = False  # Load model from file
        self.loadEpisodeFrom = 0  # Load Xth episode from file
        self.episodeCount = 40000  # Total episodes
        self.stateSize = stateSize  # Step size get from env
        self.actionSize = actionSize  # Action size get from env
        self.targetUpdateCount = 2000  # Update target model at every X step
        self.saveModelAtEvery = 10  # Save model at every X episode
        self.discountFactor = 0.99  # For qVal calculations
        self.learningRate = 0.0003  # For neural net model
        self.epsilon = 1.0  # Epsilon start value
        self.epsilonDecay = 0.99  # Epsilon decay value
        self.epsilonMin = 0.05  # Epsilon minimum value
        self.batchSize = 64  # Size of a miniBatch
        self.learnStart = 100000  # Start to train model from this step
        self.memory = deque(maxlen=200000)  # Main memory to keep batches
        self.timeOutLim = 1400  # Maximum step size for each episode
        self.savePath = '/tmp/mantisModel/'  # Model save path

        self.onlineModel = self.initNetwork()
        self.targetModel = self.initNetwork()

        self.updateTargetModel()

        # Create model file path
        try:
            os.mkdir(self.savePath)
        except Exception:
            pass

    def initNetwork(self):
        '''
        Build DNN

        return Keras DNN model
        '''
        model = Sequential()

        model.add(Dense(64, input_shape=(self.stateSize,), activation="relu", kernel_initializer="lecun_uniform"))
        model.add(Dense(64, activation="relu", kernel_initializer="lecun_uniform"))
        model.add(Dropout(0.3))
        model.add(Dense(self.actionSize, activation="linear", kernel_initializer="lecun_uniform"))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learningRate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def calcQ(self, reward, nextTarget, done):
        """
        Calculates q value
        target = reward(s,a) + gamma * max(Q(s')

        return q value in float
        """
        if done:
            return reward
        else:
            return reward + self.discountFactor * np.amax(nextTarget)

    def updateTargetModel(self):
        '''
        Update target model weights with online model weights
        '''
        self.targetModel.set_weights(self.onlineModel.get_weights())

    def calcAction(self, state):
        '''
        Caculates an Action

        returns action number in int
        '''
        
        if np.random.rand() <= self.epsilon:  # return random action
            self.qValue = np.zeros(self.actionSize)
            return random.randrange(self.actionSize)
        else:  # Ask action to neural net
            qValue = self.onlineModel.predict(state.reshape(1, self.stateSize))
            self.qValue = qValue
            return np.argmax(qValue[0])
    
    def appendMemory(self, state, action, reward, nextState, done):
        '''
        Append state to replay mem
        '''
        self.memory.append((state, action, reward, nextState, done))

    def trainModel(self, target=False):
        '''
        Train model with randomly choosen minibatches
        Uses Double DQN
        '''
        
        # Get minibatches
        miniBatch = random.sample(self.memory, self.batchSize)
        xBatch = np.empty((0, self.stateSize), dtype=np.float64)
        yBatch = np.empty((0, self.actionSize), dtype=np.float64)

        for i in range(self.batchSize):
            state = miniBatch[i][0]
            action = miniBatch[i][1]
            reward = miniBatch[i][2]
            nextState = miniBatch[i][3]
            done = miniBatch[i][4]

            qValue = self.onlineModel.predict(state.reshape(1, len(state)))
            self.qValue = qValue

            if target:
                nextTarget = self.targetModel.predict(nextState.reshape(1, len(nextState)))
            else:
                nextTarget = self.onlineModel.predict(nextState.reshape(1, len(nextState)))

            nextQValue = self.calcQ(reward, nextTarget, done)

            xBatch = np.append(xBatch, np.array([state.copy()]), axis=0)
            ySample = qValue.copy()

            ySample[0][action] = nextQValue
            yBatch = np.append(yBatch, np.array([ySample[0]]), axis=0)

            if done:
                xBatch = np.append(xBatch, np.array([nextState.copy()]), axis=0)
                yBatch = np.append(yBatch, np.array([[reward] * self.actionSize]), axis=0)

        self.onlineModel.fit(xBatch, yBatch, batch_size=self.batchSize, epochs=1, verbose=0)


class LivePlot():
    '''
    Class for live plot while training for episode and score
    '''
    def __init__(self):
        self.x = [0]
        self.y = [0]
        self.fig = plt.figure(0)
    
    def update(self, x, y, yTitle, text, updtScore=True):
        if updtScore:
            self.x.append(x)
            self.y.append(y)

        self.fig.canvas.set_window_title(text)
        plt.xlabel('Epoch', fontsize=13)
        plt.ylabel(yTitle, fontsize=13)
        plt.style.use('Solarize_Light2')
        plt.plot(self.x, self.y)
        plt.draw()
        plt.pause(0.5)
        plt.clf()


if __name__ == '__main__':
    if LIVE_PLOT:
        score_plot = LivePlot()

    env = MantisGymEnv()  # Create environment

    # get action and state sizes
    stateSize = env.stateSize
    actionSize = env.actionSize

    # Create an agent
    agent = Agent(stateSize, actionSize)

    # Load model from file if needed
    if agent.loadModel:
        agent.onlineModel.set_weights(load_model(agent.savePath+str(agent.loadEpisodeFrom)+".h5").get_weights())

        with open(agent.savePath+str(agent.loadEpisodeFrom)+'.json') as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')


    stepCounter = 0
    startTime = time.time()
    for episode in range(agent.loadEpisodeFrom + 1, agent.episodeCount):
        done = False
        state = env.reset()
        score = 0
        total_max_q = 0

        for step in range(1,999999):
            action = agent.calcAction(state)
            nextState, reward, done = env.step(action)

            if score+reward > 10000 or score+reward < -10000:
                print("Error Score is too high or too low! Resetting...")
                break

            agent.appendMemory(state, action, reward, nextState, done)

            if agent.isTrainActive and len(agent.memory) >= agent.learnStart:
                if stepCounter <= agent.targetUpdateCount:
                    agent.trainModel(False)
                else:
                    agent.trainModel(True)

            score += reward
            state = nextState

            avg_max_q_val_text = "Avg Max Q Val:{:.2f}  | ".format(np.max(agent.qValue))
            reward_text = "Reward:{:.2f}  | ".format(reward)
            action_text = "Action:{:.2f}  | ".format(action)

            inform_text = avg_max_q_val_text + reward_text + action_text
            
            if LIVE_PLOT:
                score_plot.update(episode, score, "Score", inform_text, updtScore=False)
            
            # Save model to file
            if agent.isTrainActive and episode % agent.saveModelAtEvery == 0:
                weightsPath = agent.savePath + str(episode) + '.h5'
                paramPath = agent.savePath + str(episode) + '.json'
                agent.onlineModel.save(weightsPath)
                with open(paramPath, 'w') as outfile:
                    json.dump(paramDictionary, outfile)

            total_max_q += np.max(agent.qValue)

            if (step >= agent.timeOutLim):
                print("Time out")
                done = True

            if done:
                agent.updateTargetModel()

                avg_max_q = total_max_q / step

                # Infor user
                m, s = divmod(int(time.time() - startTime), 60)
                h, m = divmod(m, 60)

                print('Ep: {} | AvgMaxQVal: {:.2f} | CScore: {:.2f} | Mem: {} | Epsilon: {:.2f} | Time: {}:{}:{}'.format(episode, avg_max_q, score, len(agent.memory), agent.epsilon, h, m, s))
                
                if LIVE_PLOT:
                    score_plot.update(episode, score, "Score", inform_text, updtScore=True)

                paramKeys = ['epsilon']
                paramValues = [agent.epsilon]
                paramDictionary = dict(zip(paramKeys, paramValues))
                break

            stepCounter += 1
            if stepCounter % agent.targetUpdateCount == 0:
                agent.updateTargetModel()

        # Epsilon decay
        if agent.epsilon > agent.epsilonMin:
            agent.epsilon *= agent.epsilonDecay

