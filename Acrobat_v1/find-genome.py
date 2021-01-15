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



runs_per_net = 4
# simulation_seconds = 60.0


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    # Run the given simulation for up to num_steps time steps.
    for runs in range(runs_per_net):
        env = gym.make("Acrobot-v1") # make new gym environment for every new agent in the epoch 

        observation = env.reset() # reset the agents observations in the newly-minted environment

        max_height = -2 # the highest point is when observation = [0 1 0 1 . .]

        done = False

        while not done:
            action = np.argmax(net.activate(observation))
        
            observation, reward, done, info = env.step(action)
        
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

            fitness = np.max([max_height, current_height])

        fitnesses.append(fitness) # if the same genome does more than one trial, this fxn will average them and return them as representative of the genomes fitness 

    # The genome's fitness is its average performance across all runs
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
