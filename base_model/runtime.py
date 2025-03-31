import keras
from brian2 import *
from brian2tools import *
import matplotlib.pyplot as plt

from utils import *
import base_model.MNISTv1p0 as model # Change to the testing model

class evalInstance:
    
    # * ======================================================================
    # * Initialization and Loading Datasets
    # * ======================================================================
    
    def __init__(self, network, num_neurons, dataset, homepath):
        self.network = network
        self.n_neurons = num_neurons
        self.path = homepath
        
        if dataset == 'mnist':
            self.n_classes = 10
            self.loadMNIST()
        
    def loadMNIST(self):
        start = time.time()
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        self.trainset = (x_train, y_train)
        self.testset =  (x_test, y_test)
        
        end = time.time()
        print('time needed to load MNIST set:', end - start)

    # * ======================================================================
    # * Importing model functions
    # * ======================================================================
    
    def inputEncoding(self, img, quant_levels=10):
        return model.temporal_switch(img, quant_levels)
    
    # * ======================================================================
    # * Network Training
    # * ======================================================================
    
    def diehl_norm_weights(self, synapse, target_weight):
        temp_conn = np.copy(synapse.w)
        temp_conn.reshape(784, self.n_neurons) # Converting to a 2D array
        
        colSums = np.sum(temp_conn, axis=0) # Calculate sum of weights for each excitatory neuron
        colFactors = target_weight/colSums
        
        for j in range(self.n_neurons):
            temp_conn[:,j] *= colFactors[j]
        temp_conn.shape = (784 * self.n_neurons)
        synapse.w = temp_conn[:]
    
    def train(self, num_examples, target_weight, single_run_time, rest_time, capture_period):
        print('starting training')
        
        result_monitor = np.zeros((num_examples, self.n_neurons))
        input_numbers = [0] * num_examples
        previous_spike_count = 0
        previous_spikes = np.zeros(self.n_neurons)
        
        j = 0
        while (j < (int(num_examples))):
            print(j/int(num_examples) *100, "% completed")
            self.diehl_norm_weights(self.network['connA2B'], target_weight) #Normalizing Weights
            
            current_img = self.trainset[0][j % len(self.trainset[0]),:,:]
            ((pixel_indices_e, times_e), (pixel_indices_i, times_i)) = self.inputEncoding(current_img) #Spike encoding
            
            self.network['spikes_in_e'].set_spikes(pixel_indices_e, times_e)
            self.network['spikes_in_i'].set_spikes(pixel_indices_i, times_i)
            
            self.network.run(single_run_time)
            
            if (j%capture_period and j > 0):
                ending = '_' + str(j)
                save_connections(self.path, self.network['connA2B'], ending)
            
            current_spike_count = self.network['B_mon'].num_spikes - previous_spike_count
            previous_spike_count = self.network['B_mon'].num_spikes
            
            result_monitor[j%num_examples,:] =  self.network['B_mon'].count[:] - previous_spikes[:]
            input_numbers[j] = self.trainset[1][j%num_examples][0]
            
            if j%1000 == 0 and j > 0:
                print('runs done', j, 'of', str((num_examples)))
            
            #Resting Time
            self.network['spikes_in_e'].set_spikes([], [])
            self.network['spikes_in_i'].set_spikes([], [])
            self.network.run(rest_time)
            
            previous_spikes = self.network['B_mon'].count[:]
            j += 1

        return 
            
