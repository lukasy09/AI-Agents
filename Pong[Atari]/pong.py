import gym
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout

""" Hyperparameters
        for our task  """

initial_games = 200
n_steps = 1000
n_test_games = 5
train_model = load_model("model_v1.h5")

env = gym.make('Pong-v0')
env.reset()
training_data = []
labels = []

def prepro(I):
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

for game in range(0, initial_games):
    game_memory = []
    prev_obs = None
    score = 0
    done = False
    while not done:
        if prev_obs is None:
            action = random.choice([2, 3])
        else:
            action = np.argmax(train_model.predict(np.asarray([prev_obs])))
        obs, reward, done, info = env.step(action)
        obs = prepro(obs)
        if prev_obs is not None:
                game_memory.append([prev_obs,  action])
        score+= reward
        prev_obs = obs
    if score >= -15:
        print("Game:" + str(game))
        print(score)
        for data in game_memory:
            training_data.append(data[0])
            labels.append(data[1])
    env.reset()
    #print(score)

    
training_data = np.asarray(training_data)    
labels = np.asarray(labels)     
labels = to_categorical(labels)


def create_model():
    model = Sequential()
    model.add(Dense(units = 128, input_dim = 6400))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 4, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
    return model

model = create_model()
model.fit(training_data, labels, epochs = 10)

model.save("model_v2.h5")

model = load_model("model_v1.h5")

for test_game in range(0, 5):
    done = False
    prev_obs = []
    print("Game:" + str(test_game))
    for step in range(0, 2000):
         env.render()
         if len(prev_obs) == 0:
             action = random.choice([2,3])
         else:
             action = np.argmax(model.predict(np.asarray([prev_obs])))
             
         obs , reward, done, info = env.step(action)
         obs = prepro(obs)
         prev_obs = obs
         if done:
             break
    env.reset()
    
