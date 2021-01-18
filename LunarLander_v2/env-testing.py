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
##  OBSERVATION/STATE: 8 floats in a flat array
##          pos.x
##          pos.y 
##          vel.x
##          vel.y
##          self.lander.angle,
##          self.lander.angularVelocity
##          1.0 if self.left_leg.ground_contact else 0.0,
##          1.0 if self.right_leg.ground_contact else 0.0
##
##  DISCRETE ACTION: one integer
##    - 0 for no action, 1 for left engine, 2 for main engine, 3 for right engine
##
##  CONTINUOUS ACTION: two floats [ main engine[-1:(0.5:1)], left-right engines[(1:-0.5):(0.5:1)] ].
##    - Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
##    - Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
##
## NOTES: - less fuel spent is better, about -30 for heuristic landing
##        -
##        -  
###################################################################################################################

env = gym.make("LunarLander-v2")


observation = env.reset() # vector (or matrix) of values...ANY INFO WE HAVE == observation

done = False

i = 0

while not done: # loop environment 

    observation, reward, done, info = env.step(0) # env.action_space.sample() is where our agent goes. for now, it is random sampling as a placeholder
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
print(env.action_space.sample())