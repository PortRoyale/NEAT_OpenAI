import os
import pickle
import neat
import gym
import numpy as np
import time


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

env = gym.make('Acrobot-v1')
observation = env. reset()

done= False

i = 0

while not done:
    action = np.argmax(net.activate(observation))

    observation, reward, done, info = env.step(action)

    i += 1

    # theta = angle of bar closest to hub
    # alpha = angle of bar farthest from hub
    #NEED TO USE ARCTAN so we can get proper sign on our angles
    theta = np.arctan(observation[1] / observation[0]) 
    alpha = np.arctan(observation[3] / observation[2]) 
    theta_deg = np.arctan(observation[1] / observation[0]) * 180 / np.pi
    alpha_deg = np.arctan(observation[3] / observation[2]) * 180 / np.pi

    gamma = 2*np.pi - theta - alpha
    
    cos_theta = observation[0]
    cos_alpha = observation[2]

    h1 = cos_theta
    h2 = np.cos(2 * np.pi - theta - alpha)

    current_height = - h1 + h2

    print(done, observation, reward, action, current_height, theta, alpha, gamma, h1, h2, i)

    env.render()

    if current_height >= 1:
        time.sleep(5)

    # time.sleep(0.2)

env.close()