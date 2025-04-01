import keras
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from brian2 import *

# * ======================================================================
# * Loading Datasets
# * ======================================================================

def load_mnist():
    start = time.time()
    trainset, testset = keras.datasets.mnist.load_data()
    end = time.time()
    
    print('time to taken loading MNIST set: ', end - start)
    return (trainset, testset)

# * ======================================================================
# * Helper Functions
# * ======================================================================

def display_mnist(pixel_arr):
    plt.imshow(pixel_arr, cmap='gray')
    plt.show()
    
def save_connections(path, synapse, ending = ''):
    connMatrix = np.copy(synapse.w)
    np.save(path + 'weights/' + ending, connMatrix)

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    plt.show()
    
