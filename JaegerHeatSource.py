# Calculates the temperature field of a quasi-static moving heat source
# Based on 
# Carslaw, H. S., and J. C. Jaeger.
# "Conduction of heat in solids, Clarendon." (1959).


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import *
from scipy.special import kn
from scipy.integrate import quad
from matplotlib import cm
import mayavi.mlab as mlab


def temperature_integration(V, n, coord_max, l):

    q_w = 17.143e6
    k = 36
    rho = 7840
    cp = 460
    alfa = k/(rho*cp)
    x = np.arange(-coord_max, coord_max, n)
    z = np.arange(-coord_max, 0, n/2)
    temperature = np.zeros([len(x), len(z)])
    err = []

    for i in range(0, len(z)):
        for j in range(0, len(x)):
            result, error = quad(lambda u:
                     1/(np.pi*k)*q_w*np.exp(-V*(x[j]-u)/(2*alfa))
                     *kn(0, V/(2*alfa)*np.sqrt((x[j]-u)**2+z[i]**2)),
                     -l/2, l/2)

            temperature[i, j] = result
            err.append(error)
    return temperature, err, x, z, alfa

def main():

    n = 0.0001
    coord_max = 0.005
    l = 0.002
    V = 0.15
    temperature, err, x, z, alpha = temperature_integration(V, n, coord_max, l)
    pe = V*l/(4*alpha)
    tempmax = np.amax(temperature)
    print("Peclet number: ", pe)
    print("Max temperature: ", tempmax)
    tlevels = [0]
    h = 50
    for i in range(1,h):
        templim = (tempmax / h)*i
        tlevels.append(templim)
        
    X, Z = np.meshgrid(x,z)
    fig = plt.figure(figsize=(5,2))
    plt.contourf(X,Z,temperature, levels = tlevels, cmap=cm.jet)
    plt.axis('off')
    plt.show()
    
main()