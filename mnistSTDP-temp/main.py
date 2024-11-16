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
n_e = 400
n_i = n_e

time_step = 1 * b.ms

if test_mode:
    dataset = (x_test, y_test)
else:
    dataset = (x_train, y_train)

v_rest_e = -65. * b.mV #resting excitatory potential
v_rest_i = -60. * b.mV #resting inhibiotry poteintial
v_reset_e = -65. * b.mV #Reset poteintial for excitatory #! Adaptive thresholding, change later
v_reset_i = -45. * b.mV #Reset poteintial for inhibitory
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV #Spiking threshold for inhibitory
refrac_e = 5. * b.ms #Refractory period for excitatory
refrac_i = 2. * b.ms #Refractory period for inhibitory

weight_ee = 78. #Synaptic weight excitatatory --> excitatory
input_intensity = 2. 
delay_ee = 10. * b.ms #Synaptic delay excitatory --> excitatory
delay_ie = 5. * b.ms #Synaptic delay inhibitory --> excitatory

neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                               : amp
        I_synI = gi * nS * (-100.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        '''

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                              : amp
        I_synI = gi * nS * (-85.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                      : 1
        dgi/dt = -gi/(2.0*ms)                                      : 1
        '''

scr_e = 'v - v_reset_e; timer = 0*ms'

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
            t_s_plus = 1 * time_step #First spike
            t_s_minus = quantized_level * time_step #Second spike
            
            #Generating bipolar spikes based on pixel sign
            row, col = divmod(i, img.shape[1])
            pixel_ind = row * img.shape[1] + col #Index of pixel
            
            if p > 0:
                positive_spike_times.append((pixel_ind, t_s_plus))
                negative_spike_times.append((pixel_ind, t_s_minus))
            else:
                positive_spike_times.append((pixel_ind, t_s_minus))
                negative_spike_times.append((pixel_ind, t_s_plus))

# Combining the spike time
spike_times = positive_spike_times + negative_spike_times

# Create SpikeGeneratorGroup for positive and negative spikes
pixel_indices = [i for i, _ in spike_times]
times = [t for _, t in spike_times]

spike_gen_group = SpikeGeneratorGroup(num_pixels, pixel_indices, times)
input_mon = SpikeMonitor(spike_gen_group)

#Preview Input Spikes
'''
run(3*b.second)
plot(input_mon.t/ms, input_mon.i, '.')
show()
'''

# * ======================================================================
# * Network population and recurrent connections
# * ======================================================================

group_ne = b.NeuronGroup(n_e, neuron_eqs_e, method='euler', threshold= v_thresh_e, refractory= refrac_e, reset= scr_e)
