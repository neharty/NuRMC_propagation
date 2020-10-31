import numpy as np
from scipy.integrate import solve_ivp
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import curve_fit

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
# fit data for permittivity tensor
epsdata = pd.read_csv('./epsdata/evals_vs_depth.csv')
print(epsdata)
zdepth = -np.array(epsdata['Nominal Depth'])
e1 = 3.157 + 0.034*np.array(epsdata['E1'])
e2 = 3.157 + 0.034*np.array(epsdata['E2'])
e3 = 3.157 + 0.034*np.array(epsdata['E3'])

e12 = (e1+e2)/2

nxy = np.sqrt(e12)
nz = np.sqrt(e3)

del e1, e2, e3, e12, epsdata

def nfit(z, a, b, c):
    z0 = z[0]
    return a + b*np.exp((z-z0)/c)

nxyparams, junk = curve_fit(nfit, zdepth, nxy)
nzparams, junk = curve_fit(nfit, zdepth, nz)

def nz(z):
    # z index of refraction function
    z0 = zdepth[0]
    a, b, c = nzparams
    return a + b*np.exp((z-z0)/c) 

def no(z):
    # x-y plane index of refraction function
    z0 = zdepth[0]
    a, b, c = nxyparams
    return a + b*np.exp((z-z0)/c)

def eps(z):
    # epsilon is diagonal
    return np.diag([(no(z))**2, (no(z))**2, (nz(z))**2])

def dnodz(z):
    #derivative of x-y index of refraction
    z0 = zdepth[0]
    a, b, c = nxyparams
    return b*np.exp((z-z0)/c)/c

def dnzdz(z):
    #derivative of z index of refraction
    z0 = zdepth[0]
    a, b, c = nzparams
    return (b/c)*np.exp((z-z0)/c)

def ns(z):
    #s-polarization index of refraction
    return no(z)

def npp(theta, z):
    #p-polarizatoin index of refraction
    return no(z)*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)

def dnpdtheta(theta, z):
    #partial of np w.r.t. theta
    return no(z)*nz(z)*np.sin(theta)*np.cos(theta)*(nz(z)**2-no(z)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def dnpdz(theta, z):
    #partial of np w.r.t. z
    return (no(z)**3*dnzdz(z)*np.sin(theta)**2+dnodz(z)*nz(z)**3*np.cos(theta)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def podes(t, y):
    # odes for p-polarization
    # form is [d(theta)/dr, dzdr]
    return [-np.cos(y[0])*dnpdz(y[0], y[1])/(npp(y[0],y[1])*np.cos(y[0])+dnpdtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0])]

def sodes(t,y):
    # odes for s-polarization
    # form is [d(theta)/dr, dzdr]
    return [-dnsdz(y[0], y[1])/(ns(y[1])), 1/np.tan(y[0])]

def pfn(theta, rmax, z0, zm, dl):
    sol = solve_ivp(podes, [0, rmax], [theta, z0], method='RK45', max_step = dl)
    zsol = sol.y[1,-1]
    return np.abs(zsol - zm)

rmax = 1000
z0 = -200
zm = -20
dl = 10

minsol = minimize(pfn, (np.pi - np.arctan((zm-z0)/rmax)), args=(rmax, z0, zm, dl))
podesol = solve_ivp(podes, [0, rmax], [minsol.x[0], z0], method='RK45', max_step = dl)

plt.plot(podesol.t, podesol.y[1], label='snells ode')
plt.plot(rmax, zm, '*', label = 'antenna')
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
