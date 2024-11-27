import matplotlib.cm as cmap
import keras
import os.path
import pickle
import brian2 as b
from struct import unpack
from brian2 import *
from brian2tools import *
from datetime import datetime
import matplotlib.pyplot as plt

# * ======================================================================
# * Functions
# * ======================================================================

def display_mnist(pixel_arr):
    plt.imshow(pixel_arr, cmap='gray')
    plt.show()

# * ======================================================================
# * Load MNIST
# * ======================================================================

start = time.time()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
end = time.time()
print('time needed to load MNIST set:', end - start)

# * ======================================================================
# * Set parameters and equations
# * ======================================================================

test_mode = False

timer_start = datetime.now()
print('Start time: ', timer_start)

ending = ''
n_input = 784
num_neurons = 400

tau = 1*b.ms
v_thresh = 10*b.mV
refrac = 5.*b.ms
v_rest = -10.*b.mV


if test_mode:
    dataset = (x_test, y_test)
else:
    dataset = (x_train, y_train)

neuron_eqs = '''
        dv/dt = (v0 - v)/tau : volt (unless refractory)
        v0 : volt
        s = s * -1 : 1
'''


# * ======================================================================
# * Creating Input Populations
# * ======================================================================

img_ind = 0 # Example Image

(img, label) = (dataset[0][img_ind], dataset[1][img_ind])
norm_img = (img - 127.5) / 127.5

quant_levels = 10 #Number of quantization levels
num_pixels = img.size #Number of pixels in MNIST (28 x 28)

positive_spike_times = []
negative_spike_times = []

for i, p in enumerate(norm_img.flatten()):
    if p != 0:
        quantized_level = int(np.floor(np.abs(p) * (quant_levels - 1)) + 1)
        if (quantized_level >= 2):
            #Generating TSC Spikes per pixel
            t_s_plus = 1 * tau #First spike
            t_s_minus = quantized_level * tau #Second spike
            
            #Generating bipolar spikes based on pixel sign
            row, col = divmod(i, img.shape[1])
            pixel_ind = row * img.shape[1] + col #Index of pixel
            
            if p > 0:
                positive_spike_times.append((pixel_ind, t_s_plus))
                negative_spike_times.append((pixel_ind, t_s_minus))
            else:
                positive_spike_times.append((pixel_ind, t_s_minus))
                negative_spike_times.append((pixel_ind, t_s_plus))

# Create SpikeGeneratorGroup for positive and negative spikes
pixel_indices_e = [i for i, _ in positive_spike_times]
times_e = [t for _, t in positive_spike_times]

pixel_indices_i = [i for i, _ in negative_spike_times]
times_i = [t for _, t in negative_spike_times]

spikes_in_e = SpikeGeneratorGroup(num_pixels, pixel_indices_e, times_e)
spikes_in_i = SpikeGeneratorGroup(num_pixels, pixel_indices_i, times_i)
print(times_e)

'''
input_mon_e = SpikeMonitor(spikes_in_e)

#Preview Input Spikes

run(3*b.second)
plot(input_mon.t/ms, input_mon.i, '.')
show()
'''

# * ======================================================================
# * Network population and recurrent connections
# * ======================================================================


neuronsA = b.NeuronGroup(num_neurons, neuron_eqs, method = 'euler', threshold=v_thresh, refractory=refrac)

neuronsA.v = v_rest
neuronsA.v0 = 0*b.mV
neuronsA.s = 1 #State variable on positive/negative spike

print("created neuron groups")

#Creating connections

conn_ein2a = b.Synapses(spikes_in_e, neuronsA, 
                        '''
                        w : 1
                        ''', 
                        on_pre='''
                        v_post += w
                        ''', method = 'exact')
conn_iin2a = b.Synapses(spikes_in_i, neuronsA, 
                        '''
                        w : 1
                        ''',
                        on_pre='''
                        v_post -= w
                        ''', method='exact')


neuronsB = b.NeuronGroup(num_neurons, neuron_eqs, method = 'euler', threshold=v_thresh, refractory=refrac)