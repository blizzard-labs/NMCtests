import matplotlib.cm as cmap
import keras
from brian2 import *
from brian2tools import *
from datetime import datetime
import matplotlib.pyplot as plt

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

# Network Parameters and Options
ending = ''
n_input = 784
num_neurons = 400
num_examples = 10000
update_interval = num_examples

# Default Clock and Timesteps
tau = 1*ms
defaultclock.dt = tau

# Loading the Dataset
if test_mode:
    dataset = (x_test, y_test)
else:
    dataset = (x_train, y_train)

# Neuron Parameters and Options
v_thresh = 10*mV
refrac = 5.*ms
v_rest = 0.*mV

neuron_eqs = '''
        v      : volt
        u      : volt/second
        v0     : volt/second
        s      : boolean
        s_pre  : boolean
'''

reset_eqs = '''
        v = v_reset
        s = False
'''

bipolarSpike = {
    'sPlus' : '(not s) and (s != s_pre)',
    'sMinus' : 's and (s != s_pre)'
}

taupre = taupost = tau
wmax = 0.01
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

stdp_eqs = '''
        w : 1
        dapre/dt = -apre/taupre    : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)  
'''

stdp_eqs_pre = '''
        apre += Apre
        w = clip(w+apost, 0, wmax)
'''

stdp_eqs_post = '''
        apost += Apost
        w = clip(w+apre, 0, wmax)
'''

# * ======================================================================
# * Creating Input Populations
# * ======================================================================

img_ind = 0 # Example Image

(img, label) = (dataset[0][img_ind], dataset[1][img_ind])
num_pixels = img.size #Number of pixels in MNIST (28 x 28)

def temporal_switch(img, quant_levels=10):
    norm_img = (img - 127.5) / 127.5

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
    
    return ((pixel_indices_e, times_e), (pixel_indices_i, times_i))

((pixel_indices_e, times_e), (pixel_indices_i, times_i)) = temporal_switch(img, 10)

spikes_in_e = SpikeGeneratorGroup(num_pixels, pixel_indices_e, times_e)
spikes_in_i = SpikeGeneratorGroup(num_pixels, pixel_indices_i, times_i)

print("created input layers")

'''
input_mon_e = SpikeMonitor(spikes_in_e)

#Preview Input Spikes

run(3*second)
plot(input_mon.t/ms, input_mon.i, '.')
show()
'''

# * ======================================================================
# * Network population and connections
# * ======================================================================
num_classes = 10 #Number of classifications (10 classes in MNIST)

A = NeuronGroup(num_neurons, neuron_eqs, method='exact', events=bipolarSpike)
B = NeuronGroup(num_neurons, neuron_eqs, method='exact', events=bipolarSpike)

neuron_groups = [A, B]

# Intialize Neuron Groups and State Variables
for g in neuron_groups:
    g.v = v_rest
    g.u = 0 *mV/ms
    g.v0 = 0 *mV/ms
    g.s = False #Threshold State
    g.s_pre = False #Previous Threshold State

print("created neuron groups")

# Neuron Updates on Timestep Intervals
@network_operation(dt=defaultclock.dt)
def neuronUpdates():
    for g in neuron_groups:
        g.u += g.v0 # Update dV/dt
        g.v += g.u * defaultclock.dt # Update v with new derivative
        g.s_pre = g.s
        g.s = (g.v >= v_thresh) #Update neuron threshold state
    
#Creating Connections
conn_e2A = Synapses(spikes_in_e, A, on_pre='v0_post = 1*mV/ms')
conn_i2A = Synapses(spikes_in_i, A, on_pre='v0_post = -1*mV/ms')

connA2B = Synapses(A, B, stdp_eqs,
                   on_pre={'plus_path' : 'v0_post = w*1*mV/ms\n' + stdp_eqs_pre,
                           'minus_path' : 'v0_post = w*-1*mV/ms\n' + stdp_eqs_pre},
                   on_event={'plus_path' : 'sPlus',
                             'minus_path' : 'sMinus'}
                   )

#! Add schedule so plus/minus pathways are scheduled before unused spike event checking

conn_e2A.connect()
conn_i2A.connect()
connA2B.connect()

print("created processing layers")

spike_monitors = {
    'A_mon' : SpikeMonitor(A),
    'B_mon' : SpikeMonitor(B)
}

print("created monitors")

# * ======================================================================
# * Output Classification
# * ======================================================================

def get_neuron_assignments(self, spike_counts, labels):
    # Assigns each neuron to the digit class that causes it to fire the most frequently
    assignments = np.zeros(self.n_neurons)
    maximum_rate = np.zeros(self.n_neurons)
    
    for digit in range(self.n_classes):
        digit_examples = np.where(labels == digit)[0]
        if len(digit_examples) > 0:
            spike_rates = np.mean(spike_counts[digit_examples], axis=0)
            
            new_assignments = np.where(spike_rates > maximum_rate)[0]
            assignments[new_assignments] = digit
            maximum_rate[new_assignments] = spike_rates[new_assignments]
    
    return assignments

def classify_spike_pattern(self, spike_counts, assignments):
    #Uses neuron assignments to classify new inputs on spike patterns
    predictions = np.zeros(self.n_classes)
    
    for digit in range(self.n_classes):
        digit_neurons = np.where(assignments == digit)[0]
        if len(digit_neurons) > 0:
            predictions[digit] = np.sum(spike_counts[:, digit_neurons]) / len(digit_neurons)
            
    return np.argsort(predictions)[::-1]
    
# * ======================================================================
# * Assembling the Network and Running Simulation
# * ======================================================================

net = Network()
net.add([spikes_in_e, spikes_in_i, A, B, 
         conn_e2A, conn_i2A, connA2B, 
         spike_monitors['A_mon'], spike_monitors['B_mon'],
         neuronUpdates])