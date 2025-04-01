import numpy as np 
from matplotlib import pyplot as plt

from scipy.special import lambertw
import math

def sgn(x):
    return abs(x) / x


def lambertInner(w):
    innerOut = -1 * (5 /w)
    print("i", innerOut)
    if (innerOut > -1 / np.e):
        return np.real(lambertw(innerOut))
    else:
        return np.real(lambertw(innerOut, -1))
    
def delta_t(w):
    return (1 / (-1 * lambertInner(w)) - 0.25)

def calcSTDPUpdate(w, eta):
    def inputKernel(x):
        return (sgn(x) * math.exp(-1 * abs(x)))
    
    return eta * inputKernel(delta_t(w))

def calcEtaUpdate(w, eta):
    return -1 * eta / 20

def logFdw(w, eta):
    pspOut = lambertInner(w)
    compOne = -1 * eta / (w * (1 + pspOut) * pspOut)
    compTwo = math.exp(-1 * abs( (-1 / pspOut) - 0.25 ))
    compThree = sgn( (-1 / pspOut) -0.25 )
    
    print("t", pspOut, compOne, compTwo, compThree)
    return math.log(abs(compOne * compTwo * compThree))

def calcLyapunovExp(w_i, eta_i, runs):
    result = [logFdw(w_i, eta_i)]
    for i in range(runs):
        w_i = w_i + calcSTDPUpdate(w_i, eta_i)
        eta_i = eta_i + calcEtaUpdate(w_i, eta_i)
        print(w_i, eta_i)
        result.append(logFdw(w_i, eta_i))

    return np.mean(result)


def generatePoints (w_range, eta_range):
    lyapunov_heatmap = np.zeros((len(w_range), len(eta_range)))
    for i, w_i in enumerate(w_range):
        for j, eta_i in enumerate(eta_range):
            val = calcLyapunovExp(w_i, eta_i, 20)
            print("wooooooooooooooohoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
            print("r", i, j, w_i, eta_i, val)
            lyapunov_heatmap[i, j] = val + 6
    
    np.savetxt("data.csv", lyapunov_heatmap, delimiter=",")
    return lyapunov_heatmap


def plot_lyapunov_heatmap(lyapunov_heatmap, w_range, eta_range):
    extent = [eta_range.min(), eta_range.max(), w_range.min(), w_range.max()]
    plt.imshow(lyapunov_heatmap, origin='lower', cmap='hot', extent=extent, aspect='equal')
    plt.colorbar(label='Lyapunov Exponent')
    plt.xlabel('w (weight)')
    plt.ylabel('eta (learning rate)')
    
    plt.show()
    
    
def read_lyapunov_heatmap_from_csv(input_file):
    lyapunov_heatmap = np.loadtxt(input_file, delimiter=',')
    return lyapunov_heatmap


resolution = 50
w_bounds = 2
eta_bounds = 2
w_range = np.linspace(-w_bounds, w_bounds, resolution)
eta_range = np.linspace(-eta_bounds, eta_bounds, resolution)

lyapunov_heatmap = generatePoints(w_range, eta_range)

input_file = "data.csv"
lyapunov_heatmap = read_lyapunov_heatmap_from_csv(input_file)
plot_lyapunov_heatmap(lyapunov_heatmap, w_range, eta_range)
