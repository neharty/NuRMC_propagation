import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from scipy.constants import speed_of_light

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

# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

#sim model parameters
def ne(z):
    # z index of refraction function
    # extraordinary index of refraction function
    return 0.9*no(z)

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
