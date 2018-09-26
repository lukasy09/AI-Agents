import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
env = gym.make("CartPole-v0")
env.reset()




def initial_population(initial_games = 100000, goal_steps = 100, score_requirement = 90):

    training_data = []
    scores = []
    accepted_scores = []
    labels = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
       
        prev_observation = []

        for _ in range(goal_steps):
         
            action = random.randrange(0,2)

            observation, reward, done, info = env.step(action)
       
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:  
                training_data.append(data[0])
                labels.append(data[1])
        env.reset()
    
        scores.append(score)
    
    training_data_save = np.array(training_data)
    np.save('cartpole-v0.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return np.array(training_data), labels

    

train_data, labels = initial_population()

labels = to_categorical(labels, 2)

model = Sequential()
model.add(Dense(units = 128, input_dim = 4))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 2, activation = 'softmax'))
    
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.fit(train_data, labels, epochs = 10)

model.save("cartpole_model.h5")
