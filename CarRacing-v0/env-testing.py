import gym
import time
import numpy as np




# Easiest continuous control task to learn from pixels, a top-down racing
# environment.
# 
# Discrete control is reasonable in this environment as well, on/off
# discretization is fine.
# 
# State consists of STATE_W x STATE_H pixels.
# 
# The reward is -0.1 every frame and +1000/N for every track tile visited, where
# N is the total number of tiles visited in the track. For example, if you have
# finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
# 
# The game is solved when the agent consistently gets 900+ points. The generated
# track is random every episode.
# 
# The episode finishes when all the tiles are visited. The car also can go
# outside of the PLAYFIELD -  that is far off the track, then it will get -100
# and die.
# 
# Some indicators are shown at the bottom of the window along with the state RGB
# buffer. From left to right: the true speed, four ABS sensors, the steering
# wheel position and gyroscope.
# 
# To play yourself (it's rather fast for humans), type:
# python gym/envs/box2d/car_racing.py
# 
# Remember it's a powerful rear-wheel drive car -  don't press the accelerator
# and turn at the same time.
#
# ACTION SPACE: 
#  self.action_space = spaces.Box(np.array([-1, 0, 0]),
#                                np.array([+1, +1, +1]),
#                                     dtype=np.float32)  # steer, gas, brake
# 
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.


env = gym.make("CarRacing-v0")


observation = env.reset() # vector (or matrix) of values...ANY INFO WE HAVE == observation

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

print(observation)
print(env.action_space)
print(env.observation_space)
# print(env.action_space.sample())