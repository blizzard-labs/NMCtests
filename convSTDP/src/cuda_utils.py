import numpy as np
from numba import *
from numba import cuda

@cuda.jit(argtypes=[uint8[:, :, :], float32[:, :, :], uint8[:, :, :], float32[:, :, :, :],
                    uint32, float32])

def conv_steps(S, V, s, w, stride, th):