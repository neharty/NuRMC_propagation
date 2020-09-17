import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L = 10

def n(z):
    return 2-(z/L)

def dndz(z):
    return -1/L

def ode(z, y):
    dydt=np.zeros(2)
    dydt[0]=y[1]/(n(z))**2
    dydt[1]=dndz(z)/n(z)
    return 

sol = solve_ivp(ode, [0,L], [1,np.sin(np.pi/3)*n(0)], method='RK23')

z = sol.y[0].T

plt.plot(sol.t, z)

plt.xlabel('t')
plt.ylabel('z')
plt.show()
