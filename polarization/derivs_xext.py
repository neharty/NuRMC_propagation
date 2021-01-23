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

def test_func(z, c):
    b = 1.78 - np.sqrt(3.157)
    return b/(1+np.exp(c*(z-z0)))

#p0 = [min(n2n3avg-n1), max(n2n3avg-n1), 1]
#p0 = [max(n2n3avg-n1), 1]
params1, p = curve_fit(test_func, depth, n2n3avg - n1)

#cont = lambda z: 1-test_func(z, *params1)
testdepths = np.linspace(0, -2800, num=280)

def ne(z):
    # x index of refraction function
    # extraordinary index of refraction function
    #cont = lambda zz: 1-test_func(zz, *params1)
    #return cont(z)*no(z)
    cont = 0.997
    return cont*no(z)

def no(z):
    # ordinary index of refraction fn
    # y-z plane index of refraction function
    # from nice8 ARA model
    
    a = nd
    b = nss - nd
    
    return a + b*np.exp(z*c)

print(ne(-2800))

#plt.plot(testdepths, no(testdepths), testdepths, ne(testdepths))
#plt.show()

def eps(z):
    # epsilon is diagonal
    return np.diag([(ne(z))**2, (no(z))**2, (no(z))**2])

def npp(z, phi, theta):
    #p-polarization index of refraction
    #return no(z)*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)
    return no(z)*ne(z)/np.sqrt((ne(z)**2*np.cos(phi)**2*np.sin(theta)**2+no(z)**2*(np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2)))

def ns(z, phi, theta):
    return no(z)

def zderiv(z, phi, theta, n, h=1e-5):
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(z, phi, theta, n, h=1e-5):
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)
