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
depth = np.array(fl['Nominal Depth'])

eperp = 3.157
deltae = 0.034

#puts eigenvals in same coords as jordan et al.
n3, n2, n1 = np.sqrt(eperp + np.array(fl['E1'])*deltae), np.sqrt(eperp + np.array(fl['E2'])*deltae), np.sqrt(eperp + np.array(fl['E3'])*deltae)

n2n3avg = (n2+n3)/2
z0 = depth[0]

def test_func(z, a,b,c):
    return a + b/(1+np.exp(-c*(z-z0)))

p0 = [min(n2n3avg-n1), max(n2n3avg-n1), 1]

params1, p = curve_fit(test_func, depth, n2n3avg - n1, p0, method='dogbox')

cont = lambda z: 1-test_func(z, *params1)

def ne(z):
    # z index of refraction function
    # extraordinary index of refraction function
    return cont(z)*no(z)

def no(z):
    # ordinary index of refraction fn
    # x-y plane index of refraction function
    # from nice8 ARA model
    
    a = nd
    b = nss - nd
    
    return a + b*np.exp(z*c)

def eps(z):
    # epsilon is diagonal
    return np.diag([(no(z))**2, (no(z))**2, (nz(z))**2])

def dnodz(z):
    #derivative of x-y index of refraction
    a = nd
    b = nss - nd
    return b*c*np.exp(z*c)

def dnedz(z):
    #derivative of z index of refraction
    a = nd
    b = nss - nd
    return cont*dnodz(z)

def ns(z):
    #s-polarization index of refraction
    return no(z)

def dnsdz(z):
    #derivative of s-polarization index of refraction
    return dnodz(z)

def npp(theta, z):
    #p-polarization index of refraction
    return no(z)*ne(z)/np.sqrt(ne(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)

def dnpdtheta(theta, z):
    #partial of np w.r.t. theta
    return no(z)*ne(z)*np.sin(theta)*np.cos(theta)*(ne(z)**2-no(z)**2)/(ne(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def dnpdz(theta, z):
    #partial of np w.r.t. z
    return (no(z)**3*dnedz(z)*np.sin(theta)**2+dnodz(z)*ne(z)**3*np.cos(theta)**2)/(ne(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def npp(z):
    return ne(z)

def ns(z):
    return no(z)

def zderiv(z, phi, theta, n, h=1e-5):
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(z, phi, theta, n, h=1e-5):
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)
