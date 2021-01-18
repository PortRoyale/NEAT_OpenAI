### this script can take as input a pickle file, and copy it with a new name ###

import pickle

old_filename = 'winner-NEAT-pickle'
copy = 'winner-NEAT-pickle-70-threshold'


# Save the winner.
with open(copy, 'wb') as perpetual_winner_filehandler: # this is the name of the new winner-pickle that will perpetuate solutions to 'find-genome.py'

    with open(old_filename, 'rb') as transient_winner_filehandler: # this winner is always rewritten over if 'find-genome.py' finds a solution
        old_winner_data = pickle.load(transient_winner_filehandler)

        pickle.dump(old_winner_data, perpetual_winner_filehandler)

        print("A copy of '{}' was saved to '{}'.".format(old_filename, copy))