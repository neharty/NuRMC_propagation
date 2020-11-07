import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light

# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

#sim model parameters
d = 1
cont = 0.9

def nz(z):
    # z index of refraction function
    a = nd
    b = nss - nd
    n1 = cont*(a + b*np.exp(-d*c))
    n2 = 1

    if  z > d:
        return 1
    if np.abs(z) <= d:
        return (n2-n1)*z/(2*d) +(n2+n1)/2
    if z < -d:
        return cont*no(z)

def no(z):
    # x-y plane index of refraction function
    # from nice8 ARA model
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if  z > d:
        return 1
    if np.abs(z) <= d:
        return (n2-n1)*z/(2*d) + (n2+n1)/2
    if z < -d:
        return a + b*np.exp(z*c)

def eps(z):
    # epsilon is diagonal
    return np.diag([(no(z))**2, (no(z))**2, (nz(z))**2])

def dnodz(z):
    #derivative of x-y index of refraction
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if z > d:
        return 0
    if np.abs(z) <= d:
        return (n2-n1)/(2*d)
    if z < -d:
        return b*c*np.exp(z*c)

def dnzdz(z):
    #derivative of z index of refraction
    a = nd
    b = nss - nd
    n1 = cont*(a + b*np.exp(-d*c))
    n2 = 1

    if z > d:
        return 0
    if np.abs(z) <= d:
        return (n2-n1)/(2*d)
    if z < -d:
        return cont*dnodz(z)

def ns(z):
    #s-polarization index of refraction
    return no(z)

def dnsdz(z):
    #derivative of s-polarization index of refraction
    return dnodz(z)

def npp(theta, z):
    #p-polarizatoin index of refraction
    return no(z)*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)

def dnpdtheta(theta, z):
    #partial of np w.r.t. theta
    if z >= -d:
        return 0
    if z < -d:
        return no(z)*nz(z)*np.sin(theta)*np.cos(theta)*(nz(z)**2-no(z)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def dnpdz(theta, z):
    #partial of np w.r.t. z
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if z > d:
        return 0
    if np.abs(z) <= d:
        return (n2-n1)/(2*d)
    if z < -d:
        return (no(z)**3*dnzdz(z)*np.sin(theta)**2+dnodz(z)*nz(z)**3*np.cos(theta)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def podes(t, y):
    # odes for p-polarization
    # form is [d(theta)/dr, dzdr]
    return [-np.cos(y[0])*dnpdz(y[0], y[1])/(npp(y[0],y[1])*np.cos(y[0])+dnpdtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0]), npp(y[0], y[1])/np.abs(np.sin(y[0]))]

def sodes(t,y):
    # odes for s-polarization
    # form is [d(theta)/dr, dzdr]
    return [-dnsdz(y[1])/(ns(y[1])), 1/np.tan(y[0]), ns(y[1])/np.abs(np.sin(y[0]))]

def objfn(theta, ode, rmax, z0, zm, dr):
    sol = solve_ivp(ode, [0, rmax], [theta, z0, 0], method='DOP853', max_step = dr)
    zsol = sol.y[1,-1]
    return zsol - zm

#rmax = 1000
#z0 = -300
#zm = -200
#dr = 10
#dz = 10

def initialangle(zd, z0):
    if zd-z0 < 0:
        return np.pi/4
    if zd-z0 >= 0:
        return np.pi/2 - np.arctan((zd-z0)/rmax)

def get_ray(minfn, odefn, mininit, rmax, z0, zm, dr, a, b):
    lb, rb = get_bounds(a, b, odefn, rmax, z0, zm, dr)
    if(lb == None and rb == None):
        return None
    else:
        minsol = root_scalar(minfn, args=(odefn, rmax, z0, zm, dr), bracket=[lb,rb])#, options={'xtol':1e-12, 'rtol':1e-12, 'maxiter':int(1e4)})

    print(minsol.converged, minsol.flag)
    odesol = solve_ivp(odefn, [0, rmax], [minsol.root, z0, 0], method='DOP853', max_step=dr)
    return odesol

def get_bounds(leftguess, rightguess, odefn, rmax, z0, zm, dr):

    if(rightguess <= leftguess):
        tmp = rightguess
        rightguess = leftguess
        leftguess = tmp
        del tmp

    xtol=1e-4
    maxiter=200

    dxi=1e-2
    dx = dxi
    zend1, zend2 = objfn(leftguess, odefn, rmax, z0, zm, dr), objfn(rightguess, odefn, rmax, z0, zm, dr)
    while np.sign(zend1) == np.sign(zend2) and rightguess <= np.pi/2:
        leftguess += dx
        rightguess += dx
        zend1, zend2 = objfn(leftguess, odefn, rmax, z0, zm, dr), objfn(rightguess, odefn, rmax, z0, zm, dr)
    if rightguess > np.pi/2:
        print('ERROR: no interval found')
        return None, None
    else:
        #tighten left bound
        inum = 0
        lastguess = leftguess
        nextguess = leftguess
        while(dx >= xtol and inum < maxiter):
            inum += 1
            nextguess = lastguess + dx
            if (np.sign(objfn(nextguess, odefn, rmax, z0, zm, dr)) != np.sign(objfn(rightguess, odefn, rmax, z0, zm, dr))):
                lastguess = nextguess
            else:
                #nextguess = lastguess
                #dx = dx/2
                break

        lb = lastguess

        #tighten right bound
        dx = dxi
        inum = 0
        lastguess = rightguess
        nextguess = rightguess
        while(dx >= xtol and inum < maxiter):
            inum += 1
            nextguess = lastguess-dx
            if np.sign(objfn(nextguess, odefn, rmax, z0, zm, dr)) != np.sign(objfn(lb, odefn, rmax, z0, zm, dr)):
                lastguess = nextguess
            else:
                #nextguess = lastguess
                #dx = dx/2
                break

        rb = lastguess

    print('returned bounds:', lb, rb)
    return lb, rb
