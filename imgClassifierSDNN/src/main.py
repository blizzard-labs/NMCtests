import numpy as np
from os.path import dirname, realpath
from math import floor

import time

def main():
    #* Flags and Retrieving data
    learn_SDNN = True
    
    if learn_SDNN:
        set_weights = False
        save_weights = True
        save_features = True
    else:
        set_weights = True
        save_weights = False
        save_features = False
        
    path = dirname(dirname(realpath(__file__)))
    spike_times_learn = [path + '/datasets/LearningSet/Face/', path + '/datasets/LearningSet/Motor/']
    spike_times_train = [path + '/datasets/TrainingSet/Face/', path + '/datasets/TrainingSet/Motor/']
    spike_times_test = [path + '/datasets/TestingSet/Face/', path + '/datasets/TestingSet/Motor/']
    
    path_set_weigths = 'results/'
    path_save_weigths = 'results/'
    path_features = 'results/'
    
    #* ----------------------------- SDNN ----------------------------------
    
    #DoG stands for Difference of Gaussions (used for comparing images)
    
    DoG_params = {'img_size': (250, 160), 'DoG_size': 7}
    
    
    
    