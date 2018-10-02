"""Imports. Working with OpenAI library"""

import gym
import numpy as np
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from keras.utils.np_utils import to_categorical


def setup_initials(env_name):
    env = gym.make(env_name)
    env.reset()
    return env
    
"""As it says, function to test what the environemt looks like and behaves.
        Just for fun or testing purposes. It doesn't really know what it is doing
        """
def random_pendulum_movements(env, n_games = 10, steps = 500):
    
    for i in range(0, n_games):
        env.reset()
        for  j in range(0, steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

def create_data(env,initial_games = 10000, goal_steps = 600, score_requirement = -650):

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
                labels.append(data[1][0])

        env.reset()
    
        scores.append(score)
        
    #np.save('pendulum.npy',training_data)

    return np.array(training_data), labels

def play_games(env,linear, poly ,n_games = 20, steps = 500):
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
                    prev_obs = prev_obs.reshape(1, -1)
                    action = linear.predict(poly.fit_transform(np.array(prev_obs)))
            choices.append(action)
            
            new_obs , reward, done, info = env.step(action)
                    
            prev_obs = new_obs
                    
            game_memory.append([new_obs, action])
                    
            score += reward
                    
            if done:
                break
    scores.append(score)
    print("Avg:", mean(scores))





""" The program is starting 
            from below lines..."""
            
env = setup_initials("Pendulum-v0")
#random_pendulum_movements(env)

data, labels = create_data(env)
labels = np.array(labels)


poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(data)
poly_reg.fit(X_poly, labels)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, labels)


play_games(env, linear = lin_reg, poly = poly_reg)





