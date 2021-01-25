import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from scipy.constants import speed_of_light
from numba import jit,njit

# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

phi = np.pi*2*np.random.random()

@jit(cache=True)
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

#sim model parameters
cont = 0.9

np.seterr(over='raise')

@jit(cache=True)
def n1(z):
    # x index of refraction function
    # extraordinary index of refraction function
    # from nice8 ARA model

    a = nd
    b = nss - nd
    return a + b*np.exp(z*c)

@jit(cache=True)
def n2(z):
    # y index of refraction function
    #same as n1 for testing

    return n1(z)

@jit(cache=True)
def n3(z):
    # z index of refraction fn
    return cont*n1(z)

def eps(z):
    # epsilon is diagonal
    return np.diag([(n1(z))**2, (n2(z))**2, (n3(z))**2])

def khat(phi, theta):
    #phi = angle from +x axis
    #theta = angle from +z axis
    return np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

def adjeps(z):
    return np.diag(np.array([n2(z)*n3(z), n1(z)*n3(z), n1(z)*n2(z)])**2)

@jit(cache=True)
def A(z, phi, theta):
    #return khat(phi, theta) @ eps(z) @ khat(phi, theta)
    return np.sin(theta)**2*(n1(z)**2*np.cos(phi)**2 + n2(z)**2*np.sin(phi)**2) + n3(z)**2*np.cos(theta)**2
    
@jit(cache=True)
def B(z, phi, theta):
    #return khat(phi, theta) @ (adjeps(z) - np.trace(adjeps(z))*np.eye(3)) @ khat(phi, theta) 
    return ((n1(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.cos(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.sin(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n3(z)**2)*np.cos(theta)**2)

@jit(cache=True)
def C(z, phi, theta):
    return (n1(z)*n2(z)*n3(z))**2

@jit(cache=True)
def ns(z, phi, theta):
    #s-polarization index of refraction
    #if z >=0:
    #    print(z)
    a = np.longdouble(A(z, phi, theta))
    b = np.longdouble(B(z, phi, theta))
    c = np.longdouble(C(z, phi, theta))
    # from https://doi.org/10.1137/1.9780898718027
    # redone in 10.1090/S0025-5718-2013-02679-8
    #w = 4*a*c
    #e = fma(-c, 4*a, w)
    #f = fma(b, b, -w)
    #discr = f + e
    discr = (b + 2*np.sqrt(a*c))*(b - 2*np.sqrt(a*c))
    return np.sqrt((b + np.sqrt(np.abs(discr)))/(2*a))
    #return n1(z)

@jit(cache=True)
def npp(z, phi, theta):
    #p-polarization index of refraction
    return np.sqrt(C(z, phi, theta)/(A(z, phi, theta)))/ns(z, phi, theta)

def zderiv(z, phi, theta, n, h=1e-5):
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(z, phi, theta, n, h=1e-5):
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)


#for testing against actual values
@jit(cache=True)
def ns_a(z, phi, theta):
    return n1(z)

@jit(cache=True)
def npp_a(z, phi, theta):
    return n1(z)*n3(z)/np.sqrt(n3(z)**2*np.cos(theta)**2+n1(z)**2*np.sin(theta)**2)



