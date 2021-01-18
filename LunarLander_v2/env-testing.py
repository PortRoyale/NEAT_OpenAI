import gym
import time
import numpy as np



###################################################################################################################
##
##   The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
## Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
## If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
## comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
## Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
## Solved is 200 points.
##
##   Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
## on its first attempt. Please see the source code for details.
##
## NOTES: - less fuel spent is better, about -30 for heuristic landing
##        -
##        -  
###################################################################################################################

env = gym.make("LunarLander-v2")


observation = env.reset() # vector (or matrix) of values...ANY INFO WE HAVE == observation

print(observation)
print(env.action_space)
print(env.observation_space)

done = False

i = 0

while not done: # loop environment 

    observation, reward, done, info = env.step(env.action_space.sample()) # env.action_space.sample() is where our agent goes. for now, it is random sampling as a placeholder
    action = env.action_space.sample()

    i += 1

    observation_rounded = np.round(observation, 3)

    print(observation_rounded, action)
    print(i, done, reward)

    env.render()

env.close()