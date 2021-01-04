import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light

# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

#sim model parameters
d = 0.01
cont = None
n2 = 1e-6

def ne(z):
    # z index of refraction function
    # extraordinary index of refraction function
    return cont*no(z)

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

def nstmp(z):
    return notmp(z)

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
