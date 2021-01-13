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
phi = None

def nz(z):
    # z index of refraction function
    return cont*no(z)

def no(z):
    # x-y plane index of refraction function
    # from nice8 ARA model
    
    a = nd
    b = nss - nd
    return a + b*np.exp(z*c)

def notmp(z):
    #for calculating the bounds
    a = nd
    b = nss - nd
    return a + b*np.exp(z*c)

def eps(z):
    # epsilon is diagonal
    return np.diag([(nz(z))**2, (no(z))**2, (no(z))**2])

def dnodz(z):
    #derivative of x-y index of refraction
    a = nd
    b = nss - nd
    return b*c*np.exp(z*c)

def dnzdz(z):
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
    #return no(z)*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)
    return no(z)*nz(z)/np.sqrt((nz(z)**2*np.cos(phi)**2*np.sin(theta)**2+no(z)**2*(np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2)))

def dnpdtheta(theta, z):
    #partial of np w.r.t. theta
    #return no(z)*nz(z)*np.sin(theta)*np.cos(theta)*(nz(z)**2-no(z)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5
    #return np.cos(phi)**2*np.cos(theta)*np.sin(theta)*nz(z)*no(z)*((2*(np.cos(theta)**2+np.sin(phi)**2*np.sin(theta)**2)*no(z)**2)-((np.cos(theta)**2 - np.cos(2*phi)*np.sin(theta)**2)*nz(z)**2))/(((np.cos(theta)**2+np.sin(phi)**2*np.sin(theta)**2)*(np.cos(phi)**2*np.sin(theta)**2*nz(z)**2 + (np.cos(theta)**2+np.sin(phi)**2*np.sin(theta)**2)*no(z)**2))**1.5)
    return - (np.cos(phi)**2*np.cos(theta)*nz(z)*no(z)*(nz(z)**2 - no(z)**2))/((nz(z)**2*np.cos(phi)**2*np.sin(theta)**2+no(z)**2*(np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2))**1.5)

def dnpdz(theta, z):
    #partial of np w.r.t. z
    #return (no(z)**3*dnzdz(z)*np.sin(theta)**2+dnodz(z)*nz(z)**3*np.cos(theta)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5
    #return ((np.cos(theta)**2+np.sin(phi)**2*np.sin(theta)**2)*no(z)**3*dnzdz(z) + np.cos(phi)**2*np.sin(theta)**2*nz(z)**3*dnodz(z))/(np.sqrt(np.cos(theta)**2+np.sin(phi)**2*np.sin(theta)**2)*((np.cos(phi)**2*np.sin(theta)**2*nz(z)**2 + (np.cos(theta)**2+np.sin(phi)**2*np.sin(theta)**2)*no(z)**2)**1.5))
    return ((np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2)*no(z)**3*dnzdz(z) + np.cos(phi)**2*np.sin(theta)**2*nz(z)**3*dnodz(z))/((nz(z)**2*np.cos(phi)**2*np.sin(theta)**2+no(z)**2*(np.sin(theta)**2*np.sin(phi)**2+np.cos(theta)**2))**1.5)    

def podes(t, y):
    # odes for p-polarization
    # form is [d(theta)/dr, dzdr, dtdr]
    return [-np.cos(y[0])*dnpdz(y[0], y[1])/(npp(y[0],y[1])*np.cos(y[0])+dnpdtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0]), npp(y[0], y[1])/np.abs(np.sin(y[0]))]

def sodes(t,y):
    # odes for s-polarization
    # form is [d(theta)/dr, dzdr, dtdr]
    return [-dnsdz(y[1])/(ns(y[1])), 1/np.tan(y[0]), ns(y[1])/np.abs(np.sin(y[0]))]

def objfn(theta, ode, event, rmax, z0, zm, dr):
    sol = shoot_ray(ode, event, 0, rmax, theta, z0, dr)
    zsol = sol.y[1,-1]
    return zsol - zm

def initialangle(zd, z0):
    if zd-z0 < 0:
        return np.pi/4
    if zd-z0 >= 0:
        return np.pi/2 - np.arctan((zd-z0)/rmax)

def get_ray(minfn, odefn, rmax, z0, zm, dr, a, b):
    lb, rb = get_bounds(a, b, odefn, rmax, z0, zm, dr)
    if(lb == None and rb == None):
        return None
    else:
        minsol = root_scalar(minfn, args=(odefn, rmax, z0, zm, dr), bracket=[lb,rb])#, options={'xtol':1e-12, 'rtol':1e-12, 'maxiter':int(1e4)})

    print(minsol.converged, minsol.flag)
    odesol = solve_ivp(odefn, [0, rmax], [minsol.root, z0, 0], method='DOP853', max_step=dr)
    return odesol

def hit_top(t, y):
    return y[1]

def hit_bot(t, y):
    return np.abs(y[1]) - 2800

def shoot_ray(odefn, event, rinit, rmax, theta0,  z0, dr):
    #event format must be in 
    sol=solve_ivp(odefn, [rinit, rmax], [theta0, z0, 0], method='DOP853', events=event, max_step=dr)
    if len(sol.t_events[0]) == 0:
        return sol
    else:
        tinit = sol.t_events[0][0]
        thetainit = sol.y_events[0][0][0]
        zinit = sol.y_events[0][0][1]
        travtime = sol.y_events[0][0][2]
        sol2 = solve_ivp(odefn, [tinit, rmax], [np.pi-thetainit, zinit, travtime], method='DOP853', events=event, max_step=dr)
        tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
        yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
        return OptimizeResult(t=tvals, y=yvals)

def get_ray_1guess(minfn, odefn, rmax, z0, zm, dr, boundguess):
    lb, rb = get_bounds_1guess(boundguess, odefn, rmax, z0, zm, dr)
    if(lb == None and rb == None):
        return None, None
    else:
        minsol = root_scalar(minfn, args=(odefn, hit_bot, rmax, z0, zm, dr), bracket=[lb,rb])#, options={'xtol':1e-12, 'rtol':1e-12, 'maxiter':int(1e4)})

    print(minsol.converged, minsol.flag)
    odesol = shoot_ray(odefn, hit_bot, 0, rmax, minsol.root, z0, dr)
    return odesol, rb

def get_bounds(leftguess, rightguess, odefn, event, rmax, z0, zm, dr, xtol = None, maxiter=None):

    if(rightguess <= leftguess):
        tmp = rightguess
        rightguess = leftguess
        leftguess = tmp
        del tmp
    
    if xtol != None:
        xtol = xtol
    else:
        xtol=1e-4

    if maxiter != None:
        maxiter = maxiter
    else:
        maxiter=200

    dxi=1e-2
    dx = dxi
    zend1, zend2 = objfn(leftguess, odefn, rmax, z0, zm, dr), objfn(rightguess, odefn, rmax, z0, zm, dr)
    while np.sign(zend1) == np.sign(zend2) and rightguess <= np.pi:
        leftguess += dx
        rightguess += dx
        zend1, zend2 = objfn(leftguess, odefn, rmax, z0, zm, dr), objfn(rightguess, odefn, rmax, z0, zm, dr)
    
    if rightguess > np.pi:
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

def get_bounds_1guess(initguess, odefn, rmax, z0, zm, dr, xtol = None, maxiter=None):

    if xtol != None:
        xtol = xtol
    else:
        xtol=1e-4

    if maxiter != None:
        maxiter = maxiter
    else:
        maxiter=200

    dxi=1e-2
    dx = dxi
    zendintl = objfn(initguess, odefn, hit_bot, rmax, z0, zm, dr)
    zendnew = zendintl
    inum = 0
    lastguess = initguess
    newguess = lastguess
    while np.sign(zendintl) == np.sign(zendnew) and newguess <= np.pi:
        lastguess = newguess
        newguess += dx
        zendnew = objfn(newguess, odefn, hit_bot, rmax, z0, zm, dr)
    
    if newguess > np.pi:
        print('ERROR: no interval found')
        return None, None
    else:
        lb, rb = lastguess, newguess
    
    print('returned bounds:', lb, rb)
    return lb, rb

