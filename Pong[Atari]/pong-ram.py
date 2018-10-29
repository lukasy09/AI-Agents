import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
env = gym.make('Pong-ram-v0')
env.reset()

training_data = []
labels = []
for i in range(0, 10000):
    score = 0
    prev_observation = []
    game_done = False;
    game_memory = []

    for game_step in range(0, 500):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])

        score += reward
        
        if score > 0:
            print("Game:" + str(i))
            print(score)
            print(prev_observation)
            training_data.append(prev_observation)
            labels.append(action)
            break
        prev_observation = observation
    env.reset()

print("End")

training_data = np.asarray(training_data)
labels = np.asarray(labels)
labels = to_categorical(labels, 6)


training_data = training_data / 255

model = Sequential()
model.add(Dense(units = 256, input_dim = 128))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 1024, activation = 'relu'))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 6, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.fit(training_data, labels, epochs = 1000)


#model.save("test-ram.h5")
model = load_model('test-ram.h5')


for test_game in range(0, 200):
    prev_obs = []
    env.reset()
    for step in range(0, 100):
        env.render()
        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.asarray([prev_obs])))
            #print(model.predict(np.asarray([prev_obs])))
            
        new_obs , reward, done, info = env.step(action)
        prev_obs = new_obs












