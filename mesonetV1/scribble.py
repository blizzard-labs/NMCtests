from brian2 import *
import utils
import numpy as np

G = NeuronGroup(2, 'v:1')
B = NeuronGroup(2, 'v:1')

S = Synapses(G, B, 'w : 1')
S.connect()

S.w[0, 0] = 1
S.w[0, 1] = 2
S.w[1, 0] = 3
S.w[1, 1] = 4

new = np.reshape(S.w, (2, 2))
print(S.w[0,:])
print(S.w)
print(new)

something = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
results = np.argwhere(something > 5)

