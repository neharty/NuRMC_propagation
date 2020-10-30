import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

d = 0.1
n1 = 1.3
n2 = 1

def n(z):
    #return 2-0.5*np.exp(z/100)
    if z < 100 - d:
        return 1
    if z >= 100 - d and z <= 100+d:
        return (n2-n1)*z/(2*d) + n2 - (100+d)*(n2-n1)/(2*d)
    if z > 100 + d:
        return n2

def dndz(z):
    #return -0.5/100 * np.exp(z/100)
    if np.abs(100-z) <= d:
        return (n2-n1)/(2*d)
    else:
        return 0

def odes(t, y):
    return [-dndz(y[1])/n(y[1]), 1/np.tan(y[0])]

sol=solve_ivp(odes, [0, 200], [0.88, 0], method='LSODA', max_step=d/10, rtol = 1e-8, atol = 1e-5)

plt.plot(sol.t, sol.y[1])
plt.xlabel('r')
plt.ylabel('z')
plt.ylim([0,200])
plt.show()
#iplt.clf()
#plt.plot(sol.y[0], sol.y[1])
#plt.show()
