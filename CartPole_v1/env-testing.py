import gym
import time


env = gym.make("CartPole-v0")
# env = gym.make("BipedalWalker-v3")

observation = env.reset() # vector (or matrix) of values...ANY INFO WE HAVE == observation

print(observation)
print(env.action_space)
print(env.observation_space)

done = False
while not done: # loop environment 
    observation, reward, done, info = env.step(env.action_space.sample()) # env.action_space.sample() is where our agent goes. for now, it is random sampling as a placeholder
    action = env.action_space.sample()

    print(action, "...", observation, "...", reward, "...", done, "...", info, "...") # ok so done == True when conditions of 