import keras
import matplotlib.pyplot as plt
from brian2 import *

'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


def display_mnist(pixel_arr):
    plt.imshow(pixel_arr, cmap='gray')
    plt.show()
    
display_mnist(x_train[0])
'''


start_scope()

v_rest_e = -65.*mV
refrac_e = 5.*ms

tau = 10*ms
eqs1 = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
'''

eqs2 = '''
    dv/dt = -v / (10*ms): volt (unless refractory)
'''

G = NeuronGroup(1, eqs2, threshold='v>0.8*volt', reset='v = 0*volt', method='euler', refractory=refrac_e)

M = StateMonitor(G, 'v', record=0)
run(50*ms)
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');
plt.show()