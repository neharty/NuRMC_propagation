import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
from scipy.interpolate import interp1d
import pandas as pd
from scipy.constants import speed_of_light
import sys
#import snell_fns as sf
#import snell_fns_x as sf
import time

import snell_prop_fns as sf
import derivs_biaxial as dv

'''
parser = argparse.ArgumentParser(description=' s and p wave simulation using snells law')
parser.add_argument('cont', type=float,
                   help='contrast in n_e from n_o')

args = parser.parse_args()
'''
'''
parser.add_argument('dl', type=float,
                   help='determines fixed track length differential size')
parser.add_argument('theta0', type=float,
                   help='determines initial zenith angle of ray')
parser.add_argument('z0', type=float,
                   help='determines starting depth of ray')

#args = parser.parse_args()

#dl = args.dl
#theta0 = args.theta0
#z0 = args.z0
'''

sf.cont = 0.9

rmax = 1000
z0 = -300
zm = -200
dr = 10
dz = 10
print(sf.cont)
cont = sf.cont

events = sf.hit_top

grazang = np.arcsin(np.trace(np.sqrt(dv.eps(0)))/(np.trace(np.sqrt(dv.eps(z0)))))
#dzg = sf.objfn(grazang, ode, rmax, z0, zm, dr)

directang = np.arctan(rmax/(np.abs(z0)-np.abs(zm)))
#dzd = sf.objfn(directang, ode, rmax, z0, zm, dr)

mirrorang = np.arctan(rmax/(np.abs(z0)+np.abs(zm)))
#dzm = sf.objfn(mirrorang, ode, rmax, z0, zm, dr)

#sortangs = np.sort(np.abs(np.array([grazang, directang, mirrorang]) - zm))

guess1 = min([grazang, directang, mirrorang])

guess2 = max([grazang, directang, mirrorang])

# from general quadratic formula

podesol1, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, guess1)
if rb is not None:
    podesol2, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, guess2)
else:
    podesol2, rb = None, None

sodesol1, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, guess1)
if rb is not None:
    sodesol2, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, guess2)
else:
    sodesol2, rb = None, None

# from analytic solns
podesol1_a, rb = sf.get_ray_1guess(sf.objfn, sf.podes_a, rmax, z0, zm, dr, guess1)
if rb is not None:
    podesol2_a, rb = sf.get_ray_1guess(sf.objfn, sf.podes_a, rmax, z0, zm, dr, guess2)
else:
    podesol2_a, rb = None, None

sodesol1_a, rb = sf.get_ray_1guess(sf.objfn, sf.sodes_a, rmax, z0, zm, dr, guess1)
if rb is not None:
    sodesol2_a, rb = sf.get_ray_1guess(sf.objfn, sf.sodes_a, rmax, z0, zm, dr, guess2)
else:
    sodesol2_a, rb = None, None

if podesol1 is not None:
    plt.plot(podesol1.t, podesol1.y[1], label = 'p1-wave')
if sodesol1 is not None:
    plt.plot(sodesol1.t, sodesol1.y[1], '--', label = 's1-wave')
if podesol2 is not None:
    plt.plot(podesol2.t, podesol2.y[1], label = 'p2-wave')
if sodesol2 is not None:
    plt.plot(sodesol2.t, sodesol2.y[1], '--', label = 's2-wave')

plt.plot(rmax, zm, 'D', label = 'antenna')
plt.plot(0, z0, '*', label = 'source')
plt.title('contrast = '+str(cont))
plt.xlabel('r')
plt.ylabel('z')
plt.legend()
#plt.savefig('snells_biaxial_example'+str(cont).replace('.','')+'.png', dpi=600)
plt.show()
plt.clf()

p1a = interp1d(podesol1_a.t, podesol1_a.y[1], 'cubic')
print('p1', max(np.abs(podesol1.y[1] - p1a(podesol1.t))))

p2a = interp1d(podesol2_a.t, podesol2_a.y[1], 'cubic')
print('p2', max(np.abs(podesol2.y[1] - p2a(podesol2.t))))

s1a = interp1d(sodesol1_a.t, sodesol1_a.y[1], 'cubic')
print('s1', max(np.abs(sodesol1.y[1] - s1a(sodesol1.t))))

s2a = interp1d(sodesol2_a.t, sodesol2_a.y[1], 'cubic')
print('s2', max(np.abs(sodesol2.y[1] - s2a(sodesol2.t))))


