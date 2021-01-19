### this script can take as input a config and a pickle file, and copy them with a new name ###

import pickle
from shutil import copyfile

file_surname = '283-fitness'



old_config = 'c:/Users/Tyler/Projects/NEATOpenAI/LunarLander_v2/config'
new_config = old_config + '-' + file_surname

old_pickle = 'winner-NEAT-pickle'
new_pickle = old_pickle + '-' + file_surname


# Copy and save a config file
copied = False
while not copied:
    copyfile(old_config, new_config)
    copied = True
    print("A copy of '{}' was saved to '{}'.".format(old_config, new_config))

# Copy and save a pickle model
with open(new_pickle, 'wb') as new_pickle_filehandler, open(old_pickle, 'rb') as old_pickle_filehandler:
    old_winner_data = pickle.load(old_pickle_filehandler)
    pickle.dump(old_winner_data, new_pickle_filehandler)

    print("A copy of '{}' was saved to '{}'.".format(old_pickle, new_pickle))
                