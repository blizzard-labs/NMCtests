from brian2 import *
from brian2tools import *
from datetime import datetime
import utils
import numpy as np

# * ======================================================================
# * Creating Input Population
# * ======================================================================

input_intensity = 2.0

def rate_coding(img, n_channels):
    rates = img.reshape(n_channels) / 8.0 * input_intensity * Hz
    return rates

def temporal_switch_coding(img, quant_levels=10, tau_encoding=1*ms):
    norm_img = (img - 127.5) / 127.5

    positive_spike_times = []
    negative_spike_times = []

    for i, p in enumerate(norm_img.flatten()):
        if p != 0:
            quantized_level = int(np.floor(np.abs(p) * (quant_levels - 1)) + 1)
            if (quantized_level >= 2):
                #Generating TSC Spikes per pixel
                t_s_plus = 1 * tau_encoding #First spike
                t_s_minus = quantized_level * tau_encoding #Second spike
                
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

# * ======================================================================
# * LIF Neuron Cores
# * ======================================================================

class LIFNeuronCore:
    def __init__(self, n_layers, layer_size, tau_membrane, v_rest, v_reset, v_thresh, tau_refractory):
        self.tau_membrane = tau_membrane
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_refractory = tau_refractory
        
        self.neuron_groups = []
        self.synapses = []
        
        self.spikeMons = {}
        self.stateMons = {}
        
        self.neuron_eqs = '''
            dv/dt = (v_rest - v + v0)/tau_membrane : volt (unless refractory)
            v0 : volt
        '''
        
        self.synapse_eqs = '''
            w : 1
            dk/dt = -k / tau_plasticity : 1 (clock-driven)
            dapre/dt = -apre / tau_pre : 1 (event-driven)
            dapost/dt = -apost / tau_post : 1 (event-driven)
        '''
        
        self.synapse_eqs_pre = '''
            apre += Apre
            w = clip(w + 0.8 * k * apost, 0, wmax)
            k = clip(((0.8 * apre)**2)/tau_k, 0, 1)
            v0_post += w
        '''
        
        self.synapse_eqs_post = '''
            apost += Apost
            w = clip(w + 0.8 * k * apre, 0, wmax)
            k = clip(((0.8 * apre)**2)/tau_k, 0, 0.5)
        '''
        
        for i in range(n_layers):
            self.neuron_groups += [NeuronGroup(layer_size, self.neuron_eqs, threshold='v>v_thresh', reset='v=v_reset', refractory=tau_refractory, method='euler')]
            self.neuron_groups[i].v = v_rest
            self.neuron_groups[i].v0 = 0*mV
        
        self.conn_map = np.empty((n_layers*layer_size, n_layers*layer_size))
        self.plast_map = np.empty((n_layers*layer_size, n_layers*layer_size))
            
    def enable_monitors(self, layer):
        self.spikeMons[layer] = SpikeMonitor(self.neuron_groups[layer], record=True)
        self.stateMons[layer] = StateMonitor(self.neuron_groups[layer], 'v', record=True)
        
        return (self.spikeMons[layer], self.stateMons[layer])

# * ======================================================================
# * Local Training and Connectivity
# * ======================================================================

def connectCore(core, connect_rule='fc', weight_rest='3*mV'):
    if connect_rule == 'fc':
        for g in range(len(core.neuron_groups) - 1):
            core.synapses += [Synapses(core.neuron_groups[g], core.neuron_groups[g+1], core.synapse_eqs, on_pre=core.synapse_eqs_pre, on_post=core.synapse_eqs_post)]
            core.synapses[-1].connect()

            #TODO: Add weight initialization here
            
def updateConnMap(core, connect_rule='fc'):
    if connect_rule == 'fc':
        for layer in range(len(core.neuron_groups) - 1):
            conn_seg = np.reshape(core.synapses[layer].w, (core.layer_size, core.layer_size)) #Reshape synapse matrix to connection map
            core.conn_map[layer*core.layer_size : (layer+1)*core.layer_size] = conn_seg #Replace segment of connection map with updated values
            
            #Same thing for plasticity
            plast_seg = np.reshape(core.synapses[layer].k, (core.layer_size, core.layer_size)) 
            core.plast_map[layer*core.layer_size : (layer+1)*core.layer_size] = plast_seg 

def highPlasticitySynapses(core, plasticity_thresh, max_layer):
    #max_layer follows 0-start indexing and is exclusive (won't include synapses starting from this layer)
    plast_seg = core.plast_map[0:max_layer*core.layer_size, 0:max_layer*core.layer_size]
    return(np.argwhere(plast_seg > plasticity_thresh)) #Returns pair of connected neurons with high plasticity
        

#TODO: Test and Debug Dijikstra's Algorithm for AI hallucinations
def DijikstrasAlg(conn_map, source, target):
    n = conn_map.shape[0]
    dist = np.full(n, np.inf)  # Initialize distances to infinity
    dist[source] = 0  # Distance to source is 0
    visited = np.full(n, False)  # Track visited nodes
    prev = np.full(n, -1)  # Track previous nodes
    
    for _ in range(n):
        min_dist = np.inf
        u = -1
        # Find the unvisited node with the smallest distance
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        if u == -1 or u == target:
            break
        
        visited[u] = True
        
        # Update distances to neighboring nodes
        for v in range(n):
            if conn_map[u, v] > 0 and not visited[v]:
                alt = dist[u] + conn_map[u, v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
    '''
    path = []
    u = target
    # Reconstruct the shortest path
    while prev[u] != -1:
        path.insert(0, u)
        u = prev[u]
    if path:
        path.insert(0, source)
    
    return path
    '''

    return dist[target]

#Accumulate external incidence and complete runs simultaneously


# * ======================================================================
# * Output Clasification
# * ======================================================================

class OutputLayer_MNIST:
    def __init__(self, connected_cores):
        self.neuroncore = LIFNeuronCore(10, 15*ms, 0*mV, -5*mV, 15*mV, 1*ms)
        for core in connected_cores:
            connect = Synapses(core.neurons, self.neuroncore.neurons, 'w : volt', on_pre='v0_post += w')
            connect.connect()
            connect.w = 'rand() * 10 * mV'
        
        self.neuroncore.enable_monitors()
    
    def get_spike_counts(self):
        return self.neuroncore.spikeMon.num_events
    
    #TODO: Update stdp rules for output layer