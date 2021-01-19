# NEAT_OpenAI

This repository is a suite of Python scripts to find efficient solutions to specific OpenAI Gym environments by configuring, training, and applying the NEAT-Python neural network framework. NeuroEvolution of Augmenting Topologies (NEAT) is a neuro-evolutionary (genetic) algorithm written about in the early 2000's by Ken Stanley. I found reading a couple of the published papers on the algorithm quite interesting, so I decided it was time to see if I could apply what I had learned to OpenAI's learning environments. I must say it is my first time working with OpenAI Gym environments, and I am extra pleased with those as well. I look forward to applying other machine learning techniques to solve these problems. I believe reading about and understanding HyperNEAT will be the next pursuit, in hopes of successfully applying what I have learned to solve the OpenAI Atari environments. Much credit goes to Sentdex on Youtube for making a NEAT and OpenAI video that I watched multiple times, and then applied what I had learned from the NEAT paper at http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf and Sentdex to other OpenAI Gym environments.

MY APPROACH:
*) Go to https://www.cs.ucf.edu/~kstanley/neat.html and learn about NEAT from the man who wrote it himself.
1) Pick an environment in this repository to test NEAT inside of.
2) Open and absorb the README in each environment folder. It will hopefully explain what the goals, rewards, and inputs/outputs should be.
3) Run env-testing.py and find out what the environment is about. In other words, look at return values of the current state of the environment. Also, understand the format of possible actions the agent can take in it's pursuit of a solution. Poke around a bit, ok. 
5) Configure 'config' file. If you don't understand the config file, go to https://neat-python.readthedocs.io/en/latest/config_file.html and also read the NEAT paper above. It helps a lot.
6) Run 'find-genome.py', it will complete if a genome that meets your config file fitness_threshold is met. If it doesn't find one in a certain amount of time or isn't showing improvement while training, you can lower your fitness_threshold or reconfigure other variables within the 'config'.
7 
    A) Check your solution with 'neat-solution.py'. Sometimes the fitness threshold will pass in training/testing/evolution but will not be valid solutions. In other words, you get a severe outlier that registers a couple lucky wins in a row (because not enough runs_per_net in 'find-solution.py').
    B) Use a config that I may have provided for a winning solution.
     NOTE: If using a solution I have found, make sure you change the 'config' and 'winner-NEAT-pickle' strings in 'neat-solution.py' to the appropriate pair of config file and pickle file. For example, 'config' would change to 'config-200-fitness' and 'winner-NEAT-pickle' would change to 'winner-NEAT-pickle-200-fitness'.
7) After a solution is found and checked, save that model and config file using 'copy-machine'. Make sure to label the surname that will be added to the copied files so you can reference it later. 
8) Go back to 5)
