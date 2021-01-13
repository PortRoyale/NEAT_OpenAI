import os
import pickle
import neat
import gym
import numpy as np


# Load the winner from thisisneat.py
with open('winner-NEAT-pickle', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome: ')
print(c)

# Load the config file, which is assumed to live in the same directory as this script. (same config used with thisisneat.py)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('MountainCar-v0')
observation = env. reset()

done= False

i = 0

while not done:
    action = np.argmax(net.activate(observation))

    observation, reward, done, info = env.step(action)

    i += 1

    print(done, observation, reward, action, i)

    env.render()

env.close()