import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

ny=1.6

L = 100

def nz(z):
    return 2-z/L

def nzp(z):
    return -1/L

def n(theta, z):
    return ny*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+ny**2*np.sin(theta)**2)

def dndtheta(theta, z):
    return ny*nz(z)*np.sin(theta)*np.cos(theta)*(nz(z)**2-ny**2)/(nz(z)**2*np.cos(theta)**2+ny**2*np.sin(theta)**2)**1.5

def dndz(theta, z):
    return ny**3*np.sin(theta)**2*nzp(z)/(nz(z)**2*np.cos(theta)**2+ny**2*np.sin(theta)**2)**1.5

def odes(t, y):
    # form is [d(theta)/dr, dzdr]
    return [-np.cos(y[0])*dndz(y[0], y[1])/(n(y[0],y[1])*np.cos(y[0])+dndtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0])]

sol=solve_ivp(odes, [0, 200], [np.pi/4, 0], method='RK45', max_step=0.1)

plt.plot(sol.t, sol.y[1])
plt.xlabel('r')
plt.ylabel('z')
plt.show()
#iplt.clf()
#plt.plot(sol.y[0], sol.y[1])
#plt.show()
