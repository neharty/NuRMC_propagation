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

cont = 0.999

import derivs_biaxial as dv

dv.cont = cont

phi = np.random.random()*2*np.pi

def podes(t, y, param='r'):
    # odes for p-polarization
    if param == 'r':
        # form is [d(theta)/dr, dzdr, dtdr], r = radial distance
        return [-np.cos(y[0])*dv.zderiv(y[1], phi, y[0], dv.npp)/(dv.npp(y[1], phi, y[0])*np.cos(y[0])+dv.thetaderiv(y[1],phi, y[0], dv.npp)*np.sin(y[0])), 1/np.tan(y[0]), dv.npp(y[1], phi, y[0])/np.abs(np.sin(y[0]))]
    if param == 'l':
        # form is [d(theta)/ds, dzds, dtds, drds]
        return [-np.sin(y[0])*np.cos(y[0])*dv.zderiv(y[1], phi, y[0], dv.npp)/(dv.npp(y[1], phi, y[0])*np.cos(y[0])+dv.thetaderiv(y[1],phi, y[0], dv.npp)*np.sin(y[0])), np.cos(y[0]), dv.npp(y[1], phi, y[0]), np.sin(y[0])]

def sodes(t, y, param='r'):
    # odes for s-polarization
    if param == 'r':
        # form is [d(theta)/dr, dzdr, dtdr], r = radial distance
        return [-np.cos(y[0])*dv.zderiv(y[1], phi, y[0], dv.ns)/(dv.ns(y[1], phi, y[0])*np.cos(y[0])+dv.thetaderiv(y[1],phi, y[0], dv.ns)*np.sin(y[0])), 1/np.tan(y[0]), dv.ns(y[1], phi, y[0])/np.abs(np.sin(y[0]))]
    if param == 'l':
        # form is [d(theta)/ds, dzds, dtds, drds]
        return [-np.sin(y[0])*np.cos(y[0])*dv.zderiv(y[1], phi, y[0], dv.ns)/(dv.ns(y[1], phi, y[0])*np.cos(y[0])+dv.thetaderiv(y[1], phi, y[0], dv.npp)*np.sin(y[0])), np.cos(y[0]), dv.ns(y[1], phi, y[0]), np.sin(y[0])]

def objfn(theta, ode, rmax, z0, zm, dr):
    #function for rootfinder
    sol = shoot_ray(ode, hit_top, 0, rmax, theta, z0, dr)
    zsol = sol.y[1,-1]
    return zsol - zm

def hit_top(t, y):
    return y[1]

hit_top.terminal = True

def hit_bot(t, y):
    return np.abs(y[1]) - 2800

def shoot_ray(odefn, event, rinit, rmax, theta0,  z0, dr):
    solver = 'RK23'
    sol=solve_ivp(odefn, [rinit, rmax], [theta0, z0, 0], method=solver, events=event)#, max_step=dr)
    if len(sol.t_events[0]) == 0:
        return sol
    else:
        tinit = sol.t_events[0][0]
        thetainit = sol.y_events[0][0][0]
        #zinit = sol.y_events[0][0][1]
        travtime = sol.y_events[0][0][2]
        sol2 = solve_ivp(odefn, [tinit, rmax], [np.pi-thetainit, 0, travtime], method=solver)#, max_step=dr)
        tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
        yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
        return OptimizeResult(t=tvals, y=yvals) 

def get_ray_1guess(minfn, odefn, rmax, z0, zm, dr, boundguess): 
    lb, rb = get_bounds_1guess(boundguess, odefn, rmax, z0, zm, dr)
    if(lb == None and rb == None):
        return None, None
    else:
        minsol = root_scalar(minfn, args=(odefn, rmax, z0, zm, dr), bracket=[lb,rb])

    print(minsol.converged, minsol.flag)
    odesol = shoot_ray(odefn, hit_top, 0, rmax, minsol.root, z0, dr)
    return odesol, rb

def get_bounds_1guess(initguess, odefn, rmax, z0, zm, dr, odeparam = 'r', xtol = None, maxiter=None):

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
    zendintl = objfn(initguess, odefn, rmax, z0, zm, dr)
    zendnew = zendintl
    inum = 0
    lastguess = initguess
    newguess = lastguess
    while np.sign(zendintl) == np.sign(zendnew) and newguess <= np.pi:
        lastguess = newguess
        newguess += dx
        zendnew = objfn(newguess, odefn, rmax, z0, zm, dr)

    if newguess > np.pi:
        print('ERROR: no interval found')
        return None, None
    else:
        lb, rb = lastguess, newguess

    print('returned bounds:', lb, rb)
    return lb, rb
