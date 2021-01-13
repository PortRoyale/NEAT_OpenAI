import gym
import time


env = gym.make("MountainCar-v0")
# env = gym.make("BipedalWalker-v3")

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

    print(action, observation, reward, done, info, i) # ok so done == True when conditions of 

    env.render()

env.close()