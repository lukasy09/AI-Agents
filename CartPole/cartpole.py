"""Imports. Working with OpenAI library"""

import gym
import numpy as np
from statistics import mean
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils.np_utils import to_categorical


#Creating environment
def create_environment():
    env = gym.make("CartPole-v1")
    env.reset()
    return env

#Returns the data that we put into network
    
def random_games(env, n_games = 40, steps = 500):
    
    for i in range(0, n_games):
        env.reset()
        
        for _ in range(steps):
             env.render()
             action = env.action_space.sample()
             observation, reward, done, info = env.step(action)
             
             if done:
                 break
        
def initial_population(env,initial_games = 10000, goal_steps = 500, score_requirement = 120):

    training_data = []
    scores = []
    accepted_scores = []
    labels = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
       
        prev_observation = []

        for _ in range(goal_steps):
         
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
       
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done:
                break
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:  
                training_data.append(data[0])
                labels.append(data[1])
        env.reset()
    
        scores.append(score)
    
    #training_data_save = np.array(training_data)
    #np.save('cartpole-v0.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    
    return np.array(training_data), labels


# Skeleton of the model
def create_model():
    model = Sequential()
    model.add(Dense(units = 128, input_dim = 4))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 2, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
    return model

#model = create_model()
#model.fit(train_data, labels, epochs = 10)

#model.save("cartpole_model.h5")


def play_games(env,model, n_games = 20, steps = 500):
    scores = []
    choices = []

    for each_game in range(n_games):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        
        for t in range(steps):
            env.render()
            
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                    action = np.argmax(model.predict(np.array([prev_obs])))
            choices.append(action)
            
            new_obs , reward, done, info = env.step(action)
                    
            prev_obs = new_obs
                    
            game_memory.append([new_obs, action])
                    
            score += reward
                    
            if done:
                break
    scores.append(score)
    print("Avg:", mean(scores))



env = create_environment()
#random_games(env)   
train_data, labels = initial_population(env)
labels = to_categorical(labels, 2)

model = load_model("cartpole_model.h5")
play_games(env ,model)