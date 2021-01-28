import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from scipy.constants import speed_of_light

# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

#sim model parameters

fl = pd.read_csv('epsdata/evals_vs_depth.csv')
depth = -np.array(fl['Nominal Depth'])

eperp = 3.157
deltae = 0.034

#puts eigenvals in same coords as jordan et al.
n3, n2, n1 = np.sqrt(eperp + np.array(fl['E1'])*deltae), np.sqrt(eperp + np.array(fl['E2'])*deltae), np.sqrt(eperp + np.array(fl['E3'])*deltae)

n2n3avg = (n2+n3)/2
z0 = depth[0]

def test_func(z, a, c):
    b = 0.002
    return b/(1+a*np.exp(c*(z-z0)))

#p0 = [3, 0.0045, 3e-2]
p0 = [3, 3e-2]
params1, p = curve_fit(test_func, depth, n2n3avg - n1, p0=p0)
print(params1)
testdepths = np.linspace(-100, -1800, num=280)

phi = 0.0

def odefns(t, y, raytype, param='r'):
    # odes 
    if raytype ==1:
        ntype = npp
    elif raytype ==2:
        ntype = ns
    else:
        raise RuntimeError('Please enter a valid ray type (1 or 2)')

    if param == 'r':
        # form is [d(theta)/dr, dzdr, dtdr], r = radial distance
        return [-np.cos(y[0])*zderiv(y[1], phi, y[0], ntype)/(ntype(y[1], phi, y[0])*np.cos(y[0])+thetaderiv(y[1],phi, y[0], ntype)*np.sin(y[0])), 1/np.tan(y[0]), ntype(y[1], phi, y[0])/np.abs(np.sin(y[0]))]
    if param == 'l':
        # form is [d(theta)/ds, dzds, dtds, drds]
        return [-np.sin(y[0])*np.cos(y[0])*zderiv(y[1], phi, y[0], ntype)/(ntype(y[1], phi, y[0])*np.cos(y[0])+thetaderiv(y[1],phi, y[0], ntype)*np.sin(y[0])), np.cos(y[0]), ntype(y[1], phi, y[0]), np.sin(y[0])]

def ne(z):
    # x index of refraction function
    # extraordinary index of refraction function
    cont = lambda zz: 1-test_func(zz, *params1)
    return cont(z)*no(z)

def no(z):
    # ordinary index of refraction fn
    # y-z plane index of refraction function
    # from nice8 ARA model
    
    a = nd
    b = nss - nd
    
    return a + b*np.exp(z*c)

print(ne(-2800))

#plt.plot(depth, n2n3avg - n1, '.', testdepths, test_func(testdepths, *params1))
#plt.show()

def eps(z):
    # epsilon is diagonal
    return np.diag([(ne(z))**2, (no(z))**2, (no(z))**2])

def npp(z, phi, theta):
    #p-polarization index of refraction
    return no(z)*ne(z)/np.sqrt((ne(z)**2*np.cos(phi)**2*np.sin(theta)**2+no(z)**2*(np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2)))

def ns(z, phi, theta):
    return no(z)

def zderiv(z, phi, theta, n, h=1e-5):
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(z, phi, theta, n, h=1e-5):
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)
