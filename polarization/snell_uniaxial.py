import numpy as np
from scipy.integrate import solve_ivp
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize

'''
parser = argparse.ArgumentParser(description='compare snells law ODE with just the regular one')
parser.add_argument('dl', type=float,
                   help='determines fixed track length differential size')
parser.add_argument('theta0', type=float,
                   help='determines initial zenith angle of ray')
parser.add_argument('z0', type=float,
                   help='determines starting depth of ray')

#args = parser.parse_args()

#dl = args.dl
#theta0 = args.theta0
#z0 = args.z0
'''

L = 100

def nz(z):
    return 2-z/L

def no(z):
    return 1.1-0.1*z/L

def eps(z):
    return np.diag([(no(z))**2, (no(z))**2, (nz(z))**2])
ny = 1.2

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

def fn(theta, r, zm, dl):
    sol = solve_ivp(odes, [0, r], [theta, 0], method='RK45', max_step = dl)
    zsol = sol.y[1,-1]
    return np.abs(zsol - zm)

minsol = minimize(fn, (np.pi/6), args=(100, 20, 0.1))
odesol = solve_ivp(odes, [0, 100], [minsol.x[0], 0], method='RK45', max_step = 0.1)

plt.plot(odesol.t, odesol.y[1], label='snells ode')
plt.plot(100, 20, '*', label = 'antenna')
plt.xlabel('r')
plt.ylabel('z')
plt.legend()

#plt.savefig('snellvprop'+str(int(np.log10(dl)))+'.pdf')
'''
fl = open('maxerr.txt', 'a')
fl.write(str(dl)+','+str(np.max(np.abs(sol.sol(r2)[1] - z2))))
fl.write('\n')
fl.close()
'''
plt.show()
