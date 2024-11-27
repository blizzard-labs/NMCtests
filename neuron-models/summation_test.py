import matplotlib.pyplot as plt
from brian2 import *


start_scope()

tau = 1 * ms

eqs = '''
    dv/dt = (0.1 * v*(I - v))/tau : 1
    I : 1
    q : 1
    '''

G = NeuronGroup(1, eqs, method='euler')
G.I = 2
G.q = 1

Se = SpikeGeneratorGroup(8, [0, 1, 2, 3, 4, 5, 6, 7], [1*ms, 2*ms, 3*ms, 4*ms, 5*ms, 11*ms, 12*ms, 13*ms])
Si = SpikeGeneratorGroup(5, [0, 1, 2, 3, 4], [6*ms, 7*ms, 8*ms, 9*ms, 10*ms])

eGconn = Synapses(Se, G, on_pre='''
                  v_post += 0.1
                  q_post = 1
                  I_post = 2
                  ''')
iGconn = Synapses(Si, G, on_pre='''
                  v_post -= 0.1
                  q_post = -1
                  I_post = 0
                  ''')

eGconn.connect()
iGconn.connect()

M = StateMonitor(G, 'v', record=0)
run(13*ms)
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');
plt.show()