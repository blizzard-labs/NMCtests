import numpy as np
import heapq
from brian2 import *
from brian2tools import *

# * ======================================================================
# * Input Encoding
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
# * Learning Algorithms: MesoNet
# * ======================================================================

# Learning rate modulation parameters
nu_decay_factor = 0.99  # Factor by which the learning rate decays
nu_min_limit = 1e-6     # Minimum limit for the learning rate
modulation_interval = 1000  # Interval (in simulation steps) to apply modulation

def variable_plasticity(step):
    global nu_ee_pre, nu_ee_post
    if step % modulation_interval == 0:  # Apply modulation at intervals
        nu_ee_pre = max(nu_ee_pre * nu_decay_factor, nu_min_limit)
        nu_ee_post = max(nu_ee_post * nu_decay_factor, nu_min_limit)
        print(f"Step {step}: Updated nu_ee_pre = {nu_ee_pre}, nu_ee_post = {nu_ee_post}")

def delta_t_similarity(delta_t_i, delta_t_j, kappa):
    """Compute similarity between two neurons based on Δt."""
    return np.exp(-abs(delta_t_i - delta_t_j) / kappa)

def build_graph(neuron_count, delta_t, kappa):
    """Build a graph where edge weights are based on Δt similarity."""
    graph = {i: [] for i in range(neuron_count)}
    for i in range(neuron_count):
        for j in range(neuron_count):
            if i != j:
                similarity = delta_t_similarity(delta_t[i], delta_t[j], kappa)
                graph[i].append((j, 1 - similarity))  # Use 1 - similarity as the cost
    return graph


# Dijkstra's algorithm for shortest paths
def dijkstra(graph, start_node):
    """Compute shortest paths from the start_node using Dijkstra's algorithm."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    paths = {node: [] for node in graph}
    paths[start_node] = [start_node]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                paths[neighbor] = paths[current_node] + [neighbor]

    return distances, paths

def attribute_weights(graph, delta_t, kappa, target_neuron, epsilon):
    """Attribute weight updates based on shortest paths and Δt similarity."""
    distances, paths = dijkstra(graph, target_neuron)
    weight_updates = np.zeros(len(graph))

    for neuron, path in paths.items():
        if neuron != target_neuron:
            similarity = delta_t_similarity(delta_t[target_neuron], delta_t[neuron], kappa)
            weight_updates[neuron] = epsilon * similarity

    return weight_updates


'''
Example usage


neuron_count = 10  # Number of neurons
delta_t = np.random.rand(neuron_count)  # Random Δt values for neurons
kappa = 0.1  # Scaling factor for Δt similarity
target_neuron = 0  # Target source neuron
epsilon = 0.01  # Learning rate for weight updates

# Build the graph and compute weight updates
graph = build_graph(neuron_count, delta_t, kappa)
weight_updates = attribute_weights(graph, delta_t, kappa, target_neuron, epsilon)

print("Weight updates:", weight_updates)
'''


# * ======================================================================
# * Clustering Algorithms
# * ======================================================================

def get_new_assignments(result_monitor, input_numbers):
    #print result_monitor.shape
    n_e = result_monitor.shape[1]
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    rate = [0] * n_e    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

def get_new_assignments_for_10(result_monitor, input_numbers, power):
    #print result_monitor.shape
    #print input_numbers.shape
    n_e = result_monitor.shape[1]
    assignments = np.zeros((n_e,10)) # initialize them as not assigned
    rate = np.zeros((10,n_e))
    count = np.zeros((10))
    for n in range(input_numbers.shape[0]):
        rate[input_numbers[n],:] += result_monitor[n,:]
        count[input_numbers[n]] += 1
    for n in range(10):
        rate[n,:] = rate[n,:] / count[n]   
    for n in range(n_e):
        rate_power = np.power(rate[:,n], power)
        if np.sum(rate_power) > 0:
            assignments[n,:] = [rate_power[i]/np.sum(rate_power) for i in range(10)]
    return assignments

def get_recognized_number_proportion(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    summed_proportion = summed_rates/ np.sum(summed_rates)
    return summed_proportion

def get_recognized_number_proportion_for_10(assignments_for_10, spike_rates):
    summed_rates = [0] * 10
    for i in range(10):
        summed_rates[i] = np.sum(spike_rates * assignments_for_10[:,i]) / len(spike_rates)
    summed_proportion = summed_rates/ np.sum(summed_rates)
    return summed_proportion

