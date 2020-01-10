import random
import numpy as np
from collections import deque
from time import sleep

from my_gazebo_turtlebot3_dqlearn import GazeboTurtlebot3DQLearnEnv

from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class Agent:
    def __init__(self, environment):
        """
        Hyperparameters definition far Agent
        """
        #  State size means state's features number so it's 360 for turtlebot
        #  360 laser point at one time
        #  So it's also dnn input layer node size
        self.stateSize = environment.laser_point_count
        #  Action space is output size and 3, F,L,R
        self.actionSize = environment.action_space

        #  Trust rate to our experiences
        self.gamma = 0.95 #  Discount
        self.alpha = 0.00025 #  Learning Rate

        #  After many experinces epsilon will be 0.01
        #  So we will do less Explore more Exploit
        self.epsilon = 1 #  Explore or Exploit
        self.epsilonDecay = 0.9999 #  Adaptive Epsilon Decay Rate
        self.epsilonMinimum = 0.01 #  Minimum for Explore

        #  Network HyperParams
        self.bias = True
        self.hiddenLayers = [150,150]
        self.activationType = "relu"
        self.regularizationFactor = 0.01

        #  Deque because after it is full 
        #  the old ones start to delete
        self.memory = deque(maxlen = 10000)

        self.dnnModel = self.buildDNN()

    #def buildDNN(self):
    #    """
    #    DNN Model Definition
    #    Stanadart DNN model
    #    """
    #    model = Sequential()
    #    model.add(Dense(12, input_dim = self.stateSize, activation="relu"))
    #    model.add(Dense(24, activation="relu"))
    #    model.add(Dense(12, activation="relu"))
    #    model.add(Dense(self.actionSize, activation="linear"))
    #    model.compile(loss = "mse", optimizer = Adam(lr = self.alpha))
    #    return model

    def buildDNN(self):
        """
        DNN Model Definition
        Stanadart DNN model
        """
        model = Sequential()
        if len(self.hiddenLayers) == 0: 
            model.add(Dense(self.actionSize, input_shape=(self.stateSize,), init='lecun_uniform', bias=self.bias))
            model.add(Activation("linear"))
        else :
            if self.regularizationFactor > 0:
                model.add(Dense(self.hiddenLayers[0], input_shape=(self.stateSize,), init='lecun_uniform', W_regularizer=l2(self.regularizationFactor),  bias=self.bias))
            else:
                model.add(Dense(self.hiddenLayers[0], input_shape=(self.stateSize,), init='lecun_uniform', bias=self.bias))

            model.add(Activation(self.activationType))
            
            for index in range(len(self.hiddenLayers)):
                layerSize = self.hiddenLayers[index]
                if self.regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(self.regularizationFactor), bias=self.bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=self.bias))

                model.add(Activation(self.activationType))

            model.add(Dense(self.actionSize, init='lecun_uniform', bias=self.bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=self.alpha, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append((state, action, reward, nextState, done))

    def act(self, state):
        """
        Get state and do action
        Exploit or Explore ???
        If explore get random action
        """
        if random.uniform(0,1) <= self.epsilon:
            return environment.randomAction()
        else:
            actValues = self.dnnModel.predict(state.reshape(1,len(state)))
            # actValues like = [[1], [2], [3]]
            # return max values indices for example 2 in here
            return np.argmax(actValues[0])

    def train(self, batchSize):
        """
        Training here
        """
        # If memory is not enough for training we pass
        if len(self.memory) < batchSize:
            return
        # Get random samples
        minibatch = random.sample(self.memory, batchSize)

        # For every samples
        for state, action, reward, nextState, done in minibatch:
            if done: # If in that sample done is true we just use reward for y_train
                target = reward
            else: # Else we use Q Learning forumlua r+gamma*max(Q(s')) for y_train
                target = reward + self.gamma*np.amax(self.dnnModel.predict(nextState.reshape(1,len(state)))[0])
            
            # So x_train = s
            # y_train = r+gamma*max(Q(s')) or if done only r
            # Remeber Q function means predict in here
            # Even we only try to max one action in here
            # We need to get all other actions predict results
            # So we can't just use np.zeros in here
            trainTarget = self.dnnModel.predict(state.reshape(1,len(state)))
            trainTarget[0][action] = target
            self.dnnModel.fit(state.reshape(1,len(state)), trainTarget,verbose=0)

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon means every episode
        we decrease the epsilon so we do less Explore
        """
        if self.epsilon > self.epsilonMinimum:
            self.epsilon *= self.epsilonDecay

def test(trainedAgent, env):
    """
    We perform test here
    """
    state = env.reset() # Reset env

    time = 0

    while True:
        env.render() # Show state visually
        action = trainedAgent.act(state) # Do action
        nextState, reward, done, info = env.step(action) # observe

        state = nextState # Update state
        time += 1
        print("Time:{} Reward:{}".format(time, reward))
        sleep(0.2)
        if done:
            print("Test Completed.")
            break

if __name__ == "__main__":
    environment = GazeboTurtlebot3DQLearnEnv() # Get env
    agent = Agent(environment) # Create Agent

    # Size of state batch which taken randomly from memory
    # We use 16 random state to fit model in every time step
    batchSize = 64
    # There will be 1000 different epsiode
    episodeNum = 1000

    for e in range(episodeNum):
        state = environment.reset()

        time = 0 # Remeber time limit is 200 after 200 system will be autoreset done will be true

        cumulativeReward = 0

        while True:
            action = agent.act(state) # Act

            nextState, reward, done, info = environment.step(action) # Observe
            
            cumulativeReward += reward

            agent.storeResults(state, action, reward, nextState, done) # Storage to mem

            state = nextState # Update State

            agent.train(batchSize) # Train with random 16 state taken from mem

            agent.adaptiveEpsilon() # Decrase epsilon

            time += 1 # Increase time

            if done:
                print("Episode:{} Time:{} Reward:{} Epsilon:{}".format(e,time, cumulativeReward, agent.epsilon))
                break

    test(agent, environment)