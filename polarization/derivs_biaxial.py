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
cont = None

def n1(z):
    # x index of refraction function
    # extraordinary index of refraction function
    # from nice8 ARA model

    a = nd
    b = nss - nd

    return a + b*np.exp(z*c)

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
    return n1(z)**2*np.cos(phi)**2*np.sin(theta)**2 + n2(z)**2*np.sin(phi)**2*np.sin(theta)**2 + n3(z)**2*np.cos(theta)**2
    

def B(z, phi, theta):
    #return khat(phi, theta) @ (adjeps(z) - np.trace(adjeps(z))*np.eye(3)) @ khat(phi, theta) 
    return ((n1(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.cos(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.sin(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n3(z)**2)*np.cos(theta)**2)

def C(z, phi, theta):
    return (n1(z)*n2(z)*n3(z))**2

def ns(z, phi, theta):
    #s-polarization index of refraction
    #if B(z,phi,theta)*B(z,phi,theta) - 4*A(z,phi,theta)*C(z,phi,theta) < 0:
        #return 1
    #else:
    #print(B(z,phi,theta)**2, 4*A(z,phi,theta)*C(z,phi,theta), B(z,phi,theta)**2 - 4*A(z,phi,theta)*C(z,phi,theta))
    #return np.sqrt((B(z,phi,theta) + np.sqrt(B(z,phi,theta)**2 - 4*A(z,phi,theta)*C(z,phi,theta)))/(2*A(z,phi,theta)))
    #discr = (B(z, phi, theta)**2/4*A(z, phi, theta)**2) - (C(z, phi, theta)/A(z,phi,theta))
    #discr = (B(z, phi, theta) - 2*np.sqrt(A(z, phi, theta)*C(z, phi, theta)))*(B(z, phi, theta) + 2*np.sqrt(A(z, phi, theta)*C(z, phi, theta)))
    #discr = 1e-6
    discr = 0
    return np.sqrt((B(z, phi, theta) + np.sqrt(discr))/(2*A(z, phi, theta)))
    #return n1(z)

def npp(z, phi, theta):
    #p-polarization index of refraction
    #if (B(z,phi,theta)*B(z,phi,theta) - 4*A(z,phi,theta)*C(z,phi,theta)) < 0:
    #    return 1
    #else:
    #return np.sqrt((B(z,phi,theta) - np.sqrt(B(z,phi,theta)*B(z,phi,theta) - 4*A(z,phi,theta)*C(z,phi,theta)))/(2*A(z,phi,theta)))
    return np.sqrt(C(z, phi, theta)/(A(z, phi, theta)))/ns(z, phi, theta)
    #return n1(z)*n3(z)/np.sqrt(n3(z)**2*np.cos(theta)**2+n1(z)**2*np.sin(theta)**2)

def zderiv(z, phi, theta, n, h=1e-11):
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(z, phi, theta, n, h=1e-11):
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)


#for testing against actual values
def ns_a(z, phi, theta):
    return n1(z)

def npp_a(z, phi, theta):
    return n1(z)*n3(z)/np.sqrt(n3(z)**2*np.cos(theta)**2+n1(z)**2*np.sin(theta)**2)



