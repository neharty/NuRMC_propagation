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
z0 = -1200
zm = -200
dr = 10
dz = 10

angs = np.linspace(np.arcsin(1/sf.ns(z0))-0.1, np.pi/2, num=200)
depths = np.linspace(-100, -2000, num=5)
zp = np.zeros(len(angs))
zs = np.zeros(len(angs))
ptests, stests = np.zeros(len(angs)), np.zeros(len(angs))

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

ctr = 0

for i in range(len(angs)):
    #for j in range(len(depths)):
        podesol = solve_ivp(sf.podes, [0, rmax], [angs[i], z0, 0], method='DOP853', max_step=dr)
        sodesol = solve_ivp(sf.sodes, [0, rmax], [angs[i], z0, 0], method='DOP853', max_step=dr)
        ptests[i] = np.abs(np.abs(sf.npp(angs[i], z0)*np.sin(angs[i])) - np.abs(sf.npp(podesol.y[0, -1], podesol.y[1, -1])*np.sin(podesol.y[0, -1])))
        stests[i] = np.abs(np.abs(sf.no(z0)*np.sin(angs[i])) - np.abs(sf.no(sodesol.y[1,-1])*np.sin(sodesol.y[0,-1])))
        #plt.figure(1)
        #plt.plot(podesol.t, podesol.y[1])
        #plt.plot
        '''
        if angs[i] > np.arcsin(sf.nss/sf.no(z0)):
            #refracted
            ax1.plot(sodesol.t, sodesol.y[1])
            ctr += 1
        elif angs[i] < np.arcsin(sf.nss/sf.no(z0)) and angs[i] >= np.arcsin(1/sf.no(z0)):
            #reflected
            ax2.plot(sodesol.t, sodesol.y[1])
            ctr += 1
        else:
            ax3.plot(sodesol.t, sodesol.y[1])
        '''
        '''
        if angs[i] > np.arctan(sf.nss*sf.cont/np.sqrt(sf.cont**2*sf.no(z0)**2-sf.nss**2)):
            #refracted
            ax1.plot(sodesol.t, sodesol.y[1])
            ctr += 1
        elif angs[i] < np.arcsin(sf.nss/sf.no(z0)) and angs[i] >= np.arcsin(1/sf.no(z0)):
            #reflected
            ax2.plot(sodesol.t, sodesol.y[1])
            ctr += 1
        else:
            ax3.plot(sodesol.t, sodesol.y[1])
        '''

        zp[i], zs[i] = podesol.y[1, -1], sodesol.y[1,-1]
'''
print('all plotted?', ctr == len(angs), ctr)
#plt.show()
plt.clf()

plt.semilogy(angs, ptests, label='p')
plt.semilogy(angs, stests, label='s')
#plt.vlines([np.arcsin(1/(sf.ns(z0)))], min(min(ptests), min(stests)), max(max(ptests), max(stests)), colors='k')
plt.legend()
#plt.show()
plt.clf()
print(np.max(ptests), np.max(stests))
'''

ax1.plot(angs, zp-zm,'-', label='p-waves')
ax2.plot(angs, zs-zm, '-', label='s-waves')
ax1.hlines(0, angs[0], angs[-1], colors='r', linestyles='-')
ax2.hlines(0, angs[0], angs[-1], colors='r', linestyles='-')
#plt.vlines([np.arcsin(1/(sf.ns(z0))), np.arcsin(sf.nss/sf.ns(z0))], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='k')
#plt.vlines([np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.no(z0)**2 - 1)), np.arctan(sf.cont*sf.nss/np.sqrt(sf.cont**2*sf.no(z0)**2 - sf.nss**2))], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')
print('npp',sf.npp(np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1)), z0))
#plt.vlines([np.arctan(sf.cont/np.sqrt(sf.no(z0)**2-1))], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='m')

plt.xlabel('initial ' + r'$\theta$')
plt.ylabel('antenna depth - final depth')
plt.title('radial distance = 1 km, initial depth = '+str(z0)+' m, contrast = ' + str(sf.cont))
plt.legend()
#plt.show()
#plt.savefig(str(z0)+'m_depthsvtheta.png', dpi=600)

#input()

print('done 1st plot')
print('p1 bounds')
print('initals:', np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
p1lb, p1rb = sf.get_bounds_1guess(np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1))+0.01, sf.podes, rmax, z0, zm, dr) 
print('p2 bounds')
p2lb, p2rb = sf.get_bounds_1guess(p1rb, sf.podes, rmax, z0, zm, dr)
if p1lb is not None and p1rb is not None:
    ax1.vlines(p1lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='orange')
    ax1.vlines(p1rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')
if p2lb is not None and p2rb is not None:
    ax1.vlines(p2lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='orange')
    ax1.vlines(p2rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')

print('s1 bounds')
s1lb, s1rb = sf.get_bounds_1guess(np.arcsin(1/sf.ns(z0))+0.01, sf.sodes, rmax, z0, zm, dr)
print('s2 bounds')
s2lb, s2rb = sf.get_bounds_1guess(s1rb, sf.sodes, rmax, z0, zm, dr)
if s1lb is not None and s1rb is not None: 
    ax2.vlines(s1lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='m')
    ax2.vlines(s1rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='c')
if s2lb is not None and s2rb is not None:
    ax2.vlines(s2lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='m')
    ax2.vlines(s2rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='c')

plt.xlabel('initial ' + r'$\theta$')
plt.ylabel('antenna depth - final depth')
plt.title('radial distance = 1 km, initial depth = '+str(z0)+' m')
plt.show()

