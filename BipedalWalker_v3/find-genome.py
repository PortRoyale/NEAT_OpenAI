"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle
import numpy as np
import neat
import visualize
import gym


runs_per_net = 3


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    # Run the given simulation for up to num_steps time steps.
    for runs in range(runs_per_net):
        env = gym.make("BipedalWalker-v3") # make new gym environment for every new agent in the epoch 

        observation = env.reset() # reset the agents observations in the newly-minted environment

        fitness = 0
        done = False

        while not done:

            action = net.activate(observation)
            observation, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness) # if the same genome does more than one trial, this fxn will add all fitnesses to an array

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) 
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-NEAT-pickle', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="winner-NEAT-fitness")
    visualize.plot_species(stats, view=True, filename="winner-NEAT-speciation")

    ### uncomment lines below to see structure of neural net and training net systems...i think?
    # node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}

    # visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                     filename="winner-NEAT")
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-NEAT-enabled.gv", show_disabled=False)
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-NEAT-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
