import numpy as np
import time
import os.path
import scipy
from functions import *
from utils import *
from mesonet import *
from brian2 import *

start = time.time()
training = get_labeled_data('./training')
end = time.time()
print ('time needed to load training set:', end - start)

#specify the location
save_path = './weights/'
load_path = './weights/'

#the time-window of simulation
single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second

#the the interval of process data and show information
progress_interval = 10
validate_interval = 5000   #no less than 2000
save_interval = 500

#number of samples for training
n_train = 60000
train_begin = 0    #specify which iteration you want the training to begin from 

net['M1'], connections['X1A1'], connections['A1A1'], neuron_groups['A1'], neuron_groups['X1'], spike_counters['A1'] = getNetwork()

#load trained weight to continue
if train_begin:
    connections['X1A1'].w = np.load(load_path + 'X1A1' + '_' + str(train_begin) + '.npy')
    neuron_groups['A1'].theta = np.load(load_path + 'theta_A1' + '_' + str(train_begin) + '.npy') *b.volt

#the intensity of rate coding
intensity_step = 0.125
start_intensity = 0.25

#the threshold of retrain
retrain_gate = np.sum([5*feature_map_size_each_kernel[kernel] for kernel in range(kernel_num)])

# run the simulation and set inputs
previous_spike_count = {}
current_spike_count = {}
assignments = {}
result_monitor = {}
results_proportion = {}
accuracy = {}

previous_spike_count['A1'] = np.zeros(neuron_num)
current_spike_count['A1'] = np.zeros(neuron_num)
assignments['A1'] = np.zeros(neuron_num)
result_monitor['A1'] = np.zeros((validate_interval,neuron_num))
results_proportion['A1'] = np.zeros((10, validate_interval))
accuracy['A1'] = []
input_numbers = np.zeros(validate_interval)

neuron_groups['X1'].rates = 0*b.hertz
net['M1'].run(0*b.second)

start = time.time()
j = train_begin
input_intensity = start_intensity
while j < n_train:

    Rates = training['x'][j%60000,:,:].reshape((n_input)) * input_intensity

    neuron_groups['X1'].rates = Rates*b.hertz
    connections['X1A1'] = normalize_weights(connections['X1A1'],norm)

    net['M1'].run(single_example_time)
    
    current_spike_count['A1'] = np.asarray(spike_counters['A1'].count[:])- previous_spike_count['A1']
    previous_spike_count['A1'] = np.copy(spike_counters['A1'].count[:])
    
    #if current_spike_count is not enough, increase the input_intensity and simulat this example again
    spike_num = np.sum(current_spike_count['A1'])
    #print spike_num

    if spike_num < retrain_gate:
        input_intensity += intensity_step
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
    else:
        result_monitor['A1'][j%validate_interval,:] = current_spike_count['A1']
        input_numbers[j%validate_interval] = training['y'][j%60000][0]

        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
        input_intensity = start_intensity

        j += 1
        
        if j % progress_interval == 0:
            print ('Progress: ', j, '/', n_train, '(', time.time() - start, 'seconds)')
            start = time.time()
            
        if j % validate_interval == 0:
            assignments['A1'] = get_new_assignments(result_monitor['A1'][:], input_numbers[:])
            test_results = np.zeros((10, validate_interval))
            for k in range(validate_interval):
                results_proportion['A1'][:,k] = get_recognized_number_proportion(assignments['A1'], result_monitor['A1'][k,:])
                test_results[:,k] = np.argsort(results_proportion['A1'][:,k])[::-1]
            difference = test_results[0,:] - input_numbers[:]
            correct = len(np.where(difference == 0)[0])
            accuracy['A1'].append(correct/float(validate_interval) * 100)
            print ('Validate accuracy: ', accuracy['A1'][-1], '(last)', np.max(accuracy['A1']), '(best)')
            
        if j % save_interval == 0:
            np.save(save_path + 'X1A1' + '_' + str(j), connections['X1A1'].w)
            np.save(save_path + 'theta_A1' + '_' + str(j), neuron_groups['A1'].theta)