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

    env.render()

    i += 1

    # theta = angle of bar closest to hub
    # alpha = angle of bar farthest from hub
    #NEED TO USE ARCTAN so we can get proper sign on our angles
    theta_tan = np.arctan(observation[1] / observation[0]) 
    alpha_tan = np.arctan(observation[3] / observation[2]) 
    theta_tan_deg = theta_tan * 180 / np.pi
    alpha_tan_deg = alpha_tan * 180 / np.pi

    theta_cos = np.arccos(observation[0])
    alpha_cos = np.arccos(observation[2])
    theta_cos_deg = theta_cos * 180 / np.pi
    alpha_cos_deg = alpha_cos * 180 / np.pi

    alpha = alpha_tan
    theta = theta_tan

    cos_theta = observation[0]

    if alpha_cos_deg >= 90:
        if theta_tan_deg > 0:            
            alpha = alpha_cos
        else:
            alpha = - alpha_cos
    if theta_cos_deg >= 90:
        if theta_tan_deg >= 0: # left quadrants
            theta = - theta_cos
        else: # right quadrants
            theta = theta_cos

    h1 = - cos_theta
    h2 = - np.cos(2 * np.pi - theta - alpha)

    current_height = h1 + h2


    print(observation[0:4], action, i)

 


env.close()