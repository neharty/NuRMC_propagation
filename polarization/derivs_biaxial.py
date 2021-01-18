import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from scipy.constants import speed_of_light
from pyfma import fma

# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

#sim model parameters
cont = None

np.seterr(over='raise')

def n1(z):
    # x index of refraction function
    # extraordinary index of refraction function
    # from nice8 ARA model

    a = nd
    b = nss - nd
    try:
        return a + b*np.exp(z*c)
    except FloatingPointError:
        print(z)

def n2(z):
    # y index of refraction function
    #same as n1 for testing

    return n1(z)

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

def A(z, phi, theta):
    #return khat(phi, theta) @ eps(z) @ khat(phi, theta)
    return np.sin(theta)**2*(n1(z)**2*np.cos(phi)**2 + n2(z)**2*np.sin(phi)**2) + n3(z)**2*np.cos(theta)**2
    
def B(z, phi, theta):
    #return khat(phi, theta) @ (adjeps(z) - np.trace(adjeps(z))*np.eye(3)) @ khat(phi, theta) 
    return ((n1(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.cos(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.sin(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n3(z)**2)*np.cos(theta)**2)

def C(z, phi, theta):
    return (n1(z)*n2(z)*n3(z))**2

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

def npp(z, phi, theta):
    #p-polarization index of refraction
    return np.sqrt(C(z, phi, theta)/(A(z, phi, theta)))/ns(z, phi, theta)

def zderiv(z, phi, theta, n, h=1e-11):
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(z, phi, theta, n, h=1e-11):
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)


#for testing against actual values
def ns_a(z, phi, theta):
    return n1(z)

def npp_a(z, phi, theta):
    return n1(z)*n3(z)/np.sqrt(n3(z)**2*np.cos(theta)**2+n1(z)**2*np.sin(theta)**2)



