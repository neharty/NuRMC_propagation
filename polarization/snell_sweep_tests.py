import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light
import snell_fns as sf

rmax = 1000
z0 = -200
zm = -200
dr = 10
dz = 10

angs = np.linspace(np.arcsin(1/sf.ns(z0))-0.1, np.pi/2, num=500)
depths = np.linspace(-100, -2000, num=5)
zp = np.zeros(len(angs))
zs = np.zeros(len(angs))

for i in range(len(angs)):
    #for j in range(len(depths)):
        podesol = solve_ivp(sf.podes, [0, rmax], [angs[i], z0, 0], method='DOP853')
        sodesol = solve_ivp(sf.sodes, [0, rmax], [angs[i], z0, 0], method='DOP853')
        #plt.figure(1)
        #plt.plot(podesol.t, podesol.y[1])
        #plt.plot
    
        zp[i], zs[i] = podesol.y[1, -1], sodesol.y[1,-1]
        
#plt.figure(2)
plt.plot(angs, zp-zm,'-', label='p-waves')
plt.plot(angs, zs-zm, '--', label='s-waves')
plt.hlines(0, angs[0], angs[-1], colors='r', linestyles='-')
plt.vlines([np.arcsin(1/(sf.ns(z0))), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax)], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='k')
plt.vlines([np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2-1))], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')

plt.xlabel('initial ' + r'$\theta$')
plt.ylabel('antenna depth - final depth')
plt.title('radial distance = 1 km, initial depth = '+str(z0)+' m, contrast = ' + str(sf.cont))
plt.legend()
#plt.savefig(str(z0)+'m_depthsvtheta.png', dpi=600)
plt.show()

print('done 1st plot')
print('p1 bounds')
print('initals:', np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
p1lb, p1rb = sf.get_bounds(np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), sf.podes, rmax, z0, zm, dr) 
print('p2 bounds')
p2lb, p2rb = sf.get_bounds(np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax), sf.podes, rmax, z0, zm, dr)
if p1lb is not None and p1rb is not None:
    plt.vlines(p1lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='orange')
    plt.vlines(p1rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')
if p2lb is not None and p2rb is not None:
    plt.vlines(p2lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='orange')
    plt.vlines(p2rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')

print('s1 bounds')
s1lb, s1rb = sf.get_bounds(np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), sf.sodes, rmax, z0, zm, dr)
print('s2 bounds')
s2lb, s2rb = sf.get_bounds(np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax), sf.sodes, rmax, z0, zm, dr)
if s1lb is not None and s1rb is not None: 
    plt.vlines(s1lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='m')
    plt.vlines(s1rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='c')
if s2lb is not None and s2rb is not None:
    plt.vlines(s2lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='m')
    plt.vlines(s2rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='c')

plt.xlabel('initial ' + r'$\theta$')
plt.ylabel('antenna depth - final depth')
plt.title('radial distance = 1 km, initial depth = '+str(z0)+' m')
plt.show()

