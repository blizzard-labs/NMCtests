from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from brian2 import *

# * ======================================================================
# * Helper Functions
# * ======================================================================

def display_mnist(pixel_arr):
    plt.imshow(pixel_arr, cmap='gray')
    plt.show()
    
def save_connections(path, synapse, ending = ''):
    connMatrix = np.copy(synapse.w)
    np.save(path + 'weights/' + ending, connMatrix)
