#Summation (Linear) Neuron with threshold balancing


import matplotlib.pyplot as plt
from brian2 import *

start_scope()

defaultclock.dt = 1 * ms

eqs = '''
    v : volt
    u : volt/second
    v0 : volt/second
    '''

G = NeuronGroup(1, eqs, method='exact')

G.v = 0* mV
G.u = 0* mV/ms
G.v0 = 0 *mV/ms

@network_operation(dt=defaultclock.dt)
def update_u():
    G.u += G.v0
    G.v += G.u * defaultclock.dt

Se = SpikeGeneratorGroup(5, [0, 1, 2, 3, 4], [1*ms, 2*ms, 3*ms, 4*ms, 5*ms])
Si = SpikeGeneratorGroup(5, [0, 1, 2, 3, 4], [6*ms, 7*ms, 8*ms, 9*ms, 10*ms])

eGconn = Synapses(Se, G, on_pre='v0_post = 1*mV/ms')
iGconn = Synapses(Si, G, on_pre='v0_post = -1*mV/ms')

eGconn.connect()
iGconn.connect()

M = StateMonitor(G, 'v', record=0)
run(13*ms)
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');
plt.show()