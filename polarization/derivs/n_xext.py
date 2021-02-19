import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from scipy.constants import speed_of_light

class n_xext:
    # ice parameters
    nss = 1.35
    nd = 1.78
    c = 0.0132 

    def __init__(self, ntype):
        self.ntype = ntype
    
    def n(self, z, phi, theta):
        if self.ntype == 1:
            return self.n1(z, phi, theta)
        if self.ntype == 2:
            return self.n2(z, phi, theta)
        else:
            return RuntimeError('please enter a valid n type (1 or 2)')
    
    def ne(z):
        # x index of refraction function
        # extraordinary index of refraction function
        return cont(z)*no(z)

    def no(z):
        # ordinary index of refraction fn
        # y-z plane index of refraction function
        # from nice8 ARA model
        
        a = nd
        b = nss - nd
        
        return a + b*np.exp(z*c)


    def n1(z, phi, theta):
        #p-polarization index of refraction
        return no(z)*ne(z)/np.sqrt((ne(z)**2*np.cos(phi)**2*np.sin(theta)**2+no(z)**2*(np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2)))

    def n2(z, phi, theta):
        return no(z)

    def zderiv(z, phi, theta, n, h=1e-5):
        return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

    def thetaderiv(z, phi, theta, n, h=1e-5):
        return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)
