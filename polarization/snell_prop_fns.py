import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light
import importlib

#derivfile  = str(input('Enter derivatives file: '))
#dv = importlib.import_module(derivfile)

cont = 0.9

import derivs_zext as dv

dv.cont = cont

def podes(t, y, param='r'):
    # odes for p-polarization
    if param == 'r':
        # form is [d(theta)/dr, dzdr, dtdr]
        return [-np.cos(y[0])*dv.dnpdz(y[0], y[1])/(dv.npp(y[0],y[1])*np.cos(y[0])+dv.dnpdtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0]), dv.npp(y[0], y[1])/np.abs(np.sin(y[0]))]
    if param == 'l':
        # form is [d(theta)/ds, dzds, dtds, drds]
        return [-np.sin(y[0])*np.cos(y[0])*dv.dnpdz(y[0], y[1])/(dv.npp(y[0],y[1])*np.cos(y[0])+dv.dnpdtheta(y[0],y[1])*np.sin(y[0])), np.cos(y[0]), dv.npp(y[0], y[1]), np.sin(y[1])]

def sodes(t, y, param='r'):
    # odes for s-polarization
    if param == 'r':
        # form is [d(theta)/dr, dzdr, dtdr]
        return [-dv.dnsdz(y[1])/(dv.ns(y[1])), 1/np.tan(y[0]), dv.ns(y[1])/np.abs(np.sin(y[0]))]
    if param == 'l':
        # form is [d(theta)/ds, dzds, dtds, drds]
        return [-np.sin(y[0])*dv.dnsdz(y[1])/(dv.ns(y[1])), np.cos(y[0]), dv.ns(y[1]), np.sin(y[0])]

def objfn(theta, ode, event, rmax, z0, zm, dr):
    #function for rootfinder
    sol = shoot_ray(ode, event, 0, rmax, theta, z0, dr)
    zsol = sol.y[1,-1]
    return zsol - zm

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
    sol=solve_ivp(odefn, [rinit, rmax], [theta0, z0, 0], method='DOP853', events=event, max_step=dr)
    if len(sol.t_events[0]) == 0:
        return sol
    elif sol.t_events[0] is not None:
        tinit = sol.t_events[0][0]
        thetainit = sol.y_events[0][0][0]
        zinit = sol.y_events[0][0][1]
        travtime = sol.y_events[0][0][2]
        sol2 = solve_ivp(odefn, [tinit, rmax], [np.pi-thetainit, zinit, travtime], method='DOP853', events=event, max_step=dr)
        tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
        yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
        return OptimizeResult(t=tvals, y=yvals)
    else:
        tinit = sol.t_events[0][1]
        thetainit = sol.y_events[0][1][0]
        zinit = sol.y_events[0][1][1]
        travtime = sol.y_events[0][1][2]
        sol2 = solve_ivp(odefn, [tinit, rmax], [np.pi-thetainit, zinit, travtime], method='DOP853', events=event, max_step=dr)
        tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
        yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
        return OptimizeResult(t=tvals, y=yvals)


def get_ray_1guess(minfn, odefn, events, rmax, z0, zm, dr, boundguess):
    lb, rb = get_bounds_1guess(boundguess, odefn, events, rmax, z0, zm, dr)
    if(lb == None and rb == None):
        return None, None
    else:
        minsol = root_scalar(minfn, args=(odefn, events, rmax, z0, zm, dr), bracket=[lb,rb])#, options={'xtol':1e-12, 'rtol':1e-12, 'maxiter':int(1e4)})

    print(minsol.converged, minsol.flag)
    odesol = shoot_ray(odefn, events, 0, rmax, minsol.root, z0, dr)
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

def get_bounds_1guess(initguess, odefn, events, rmax, z0, zm, dr, odeparam = 'r', xtol = None, maxiter=None):

    if xtol != None:
        xtol = xtol
    else:
        xtol=1e-4

    if maxiter != None:
        maxiter = maxiter
    else:
        maxiter=200
    
    param = odeparam

    dxi=1e-2
    dx = dxi
    zendintl = objfn(initguess, odefn, events, rmax, z0, zm, dr)
    zendnew = zendintl
    inum = 0
    lastguess = initguess
    newguess = lastguess
    while np.sign(zendintl) == np.sign(zendnew) and newguess <= np.pi:
        lastguess = newguess
        newguess += dx
        zendnew = objfn(newguess, odefn, events, rmax, z0, zm, dr)

    if newguess > np.pi:
        print('ERROR: no interval found')
        return None, None
    else:
        lb, rb = lastguess, newguess

    print('returned bounds:', lb, rb)
    return lb, rb

