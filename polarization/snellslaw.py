import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

L = 100

def n(z):
    return 2-z/L

def dndz(z):
    return -1/L

def odes(t, y):
    # form is [d(theta)/dr, dzdr]
    return [-dndz(y[1])/n(y[1]), 1/np.tan(y[0])]

sol=solve_ivp(odes, [0, 200], [np.pi/4, 0], method='RK45', max_step=0.1)

plt.plot(sol.t, sol.y[1])
plt.xlabel('r')
plt.ylabel('z')
plt.show()
#iplt.clf()
#plt.plot(sol.y[0], sol.y[1])
#plt.show()
