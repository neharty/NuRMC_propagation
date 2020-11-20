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
z0 = -400
zm = -200
dr = 10
dz = 10

def pguess(z0):
    if np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1)) < np.pi/2-np.arctan((-zm-z0)/rmax):
        return np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1))
    else:
        return np.pi/2-np.arctan((-zm-z0)/rmax)

def sguess(z0):
    if np.arcsin(1/sf.ns(z0)) < np.pi/2-np.arctan((-zm-z0)/rmax):
        return np.arcsin(1/sf.ns(z0))
    else:
        return np.pi/2-np.arctan((-zm-z0)/rmax)

def guess(z0):
    return min(np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1)), np.pi/2-np.arctan((-zm-z0)/rmax), np.arcsin(1/sf.ns(z0)))

angs = np.linspace(np.arcsin(1/sf.ns(z0))-0.1, np.pi/2, num=1000)
enddepths = np.zeros(len(angs))
depths = np.linspace(-100, -2000, num=5)
zp = np.zeros(len(angs))
zs = np.zeros(len(angs))
ptests, stests = np.zeros(len(angs)), np.zeros(len(angs))

# shooting different ray plots
#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()

for i in range(len(angs)):
    podesol = sf.shoot_ray(sf.podes, sf.hit_top, 0, rmax, angs[i], z0, dr)
    sodesol = sf.shoot_ray(sf.sodes, sf.hit_top, 0, rmax, angs[i], z0, dr)
    ptests[i] = np.abs(np.abs(sf.npp(angs[i], z0)*np.sin(angs[i])) - np.abs(sf.npp(podesol.y[0, -1], podesol.y[1, -1])*np.sin(podesol.y[0, -1])))
    stests[i] = np.abs(np.abs(sf.no(z0)*np.sin(angs[i])) - np.abs(sf.no(sodesol.y[1,-1])*np.sin(sodesol.y[0,-1])))
    enddepths[i] = max(podesol.t[-1], sodesol.t[-1])
    #ax1.plot(podesol.t, podesol.y[1])
    #ax2.plot(sodesol.t, sodesol.y[1])

    zp[i], zs[i] = podesol.y[1, -1], sodesol.y[1,-1]
'''
ax1.set_xlabel('radial distance [m]')
ax1.set_ylabel('depth [m]')
ax1.set_title('p-waves\nradial distance = 1 km, initial depth = '+str(z0)+' m, contrast = ' + str(sf.cont))
fig1.tight_layout()
fig1.savefig('p-raysweep'+str(z0)+'_'+str(sf.cont).replace('.','')+'.png', dpi = 600)

ax2.set_xlabel('radial distance')
ax2.set_ylabel('depth [m]')
ax2.set_title('s-waves\nradial distance = 1 km, initial depth = '+str(z0)+' m, contrast = ' + str(sf.cont))
fig2.tight_layout()
fig2.savefig('s-raysweep'+str(z0)+'_'+str(sf.cont).replace('.','')+'.png', dpi = 600)
'''
# theta vs final depth plot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

ax1.plot(angs, zp,'-', label='p-waves')
ax2.plot(angs, zs, '-', label='s-waves')
ax1.hlines(zm, angs[0], angs[-1], colors='r', linestyles='-', label='antenna depth')
ax1.vlines(np.arctan(sf.cont*sf.no(0)/np.sqrt(sf.no(z0)**2-sf.no(0)**2)),min(min(zp),min(zs)), 0)
ax2.hlines(zm, angs[0], angs[-1], colors='r', linestyles='-', label= 'antenna depth')
ax2.vlines(np.arcsin(sf.ns(0)/sf.ns(z0)), min(min(zp),min(zs)), 0)
print('npp',sf.npp(np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1)), z0))

fig.suptitle('radial distance = 1 km, initial depth = '+str(z0)+' m, contrast = ' + str(sf.cont))
ax1.legend()
ax1.set_ylabel('ray depth at 1 km')
ax2.set_xlabel('initial angle')
ax2.legend()

#plt.show()
#plt.savefig(str(z0)+'m_depthsvtheta.png', dpi=600)

#input()

print('done 1st plot')
print('p1 bounds')
p1lb, p1rb = sf.get_bounds_1guess(0.1, sf.podes, rmax, z0, zm, dr)
#print(sf.shoot_ray(sf.podes, sf.hit_top, 0, rmax, p1lb, z0, dr).y[1,-1], sf.shoot_ray(sf.podes, sf.hit_top, 0, rmax, p1rb, z0, dr).y[1,-1], sf.shoot_ray(sf.podes, sf.hit_top, 0, rmax, (p1rb+p1lb)/2, z0, dr).y[1,-1])
print('p2 bounds')
if p1rb is not None:
    p2lb, p2rb = sf.get_bounds_1guess(p1rb, sf.podes, rmax, z0, zm, dr)
else:
    p2lb, p2rb = None, None
if p1lb is not None and p1rb is not None:
    ax1.vlines(p1lb, min(min(zp), min(zs)), 0, colors='orange')
    ax1.vlines(p1rb, min(min(zp), min(zs)), 0, colors='g')
if p2lb is not None and p2rb is not None:
    ax1.vlines(p2lb, min(min(zp), min(zs)), 0, colors='orange')
    ax1.vlines(p2rb, min(min(zp), min(zs)), 0, colors='g')

print('s1 bounds')
s1lb, s1rb = sf.get_bounds_1guess(0.1, sf.sodes, rmax, z0, zm, dr)
print('s2 bounds')
if s1rb is not None:
    s2lb, s2rb = sf.get_bounds_1guess(s1rb, sf.sodes, rmax, z0, zm, dr)
else:
    s2lb, s2rb = None, None
if s1lb is not None and s1rb is not None: 
    ax2.vlines(s1lb, min(min(zp), min(zs)), 0, colors='m')
    ax2.vlines(s1rb, min(min(zp), min(zs)), 0, colors='c')
if s2lb is not None and s2rb is not None:
    ax2.vlines(s2lb, min(min(zp), min(zs)), 0, colors='m')
    ax2.vlines(s2rb, min(min(zp), min(zs)), 0, colors='c')


fig.tight_layout(rect=[0, 0.03, 1, 0.95])
print(max(np.abs(enddepths - rmax)))
plt.show()
#fig.savefig('rootexample'+str(z0)+'_'+str(sf.cont).replace('.','')+'.png', dpi = 600)

