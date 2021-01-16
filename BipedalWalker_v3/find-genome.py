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



runs_per_net = 2
# simulation_seconds = 60.0


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    # Run the given simulation for up to num_steps time steps.
    for runs in range(runs_per_net):
        env = gym.make("BipedalWalker-v3") # make new gym environment for every new agent in the epoch 

        observation = env.reset() # reset the agents observations in the newly-minted environment


        done = False

        fitness = 0 # can only go 100 to the left before fall off cliff

        i = 0 # add transience to fitness

        while not done:
            action = net.activate(observation)
        
            observation, reward, done, info = env.step(action)

            i += 1
        
            # transient_factor = i / 1600 # (.995) ** 1600 = 0.003. 1600 is cycle loop limit. will be 1.0 on first frame and 0 at last frame
            

            if reward > 0:
                fitness += (2*reward)
            else:    
                fitness += reward


            # fitness = np.max([max_distance_travelled, reward])

            
            
            # fitness = reward




        fitnesses.append(fitness) # if the same genome does more than one trial, this fxn will add all fitnesses to an array


    # print(fitnesses, np.mean(fitnesses))

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
