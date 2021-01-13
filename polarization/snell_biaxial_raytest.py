import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light
import sys
#import snell_fns as sf
#import snell_fns_x as sf
import time

import snell_prop_fns as sf

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

sf.cont = 0.999

rmax = 1000
z0 = -300
zm = -200
dr = 10
dz = 10
print(sf.cont)
cont = sf.cont

events = sf.hit_top

rguess = 0.1

podesol1, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, rguess)
if rb is not None:
    podesol2, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, rb)
else:
    podesol2, rb = None, None

sodesol1, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, rguess)
if rb is not None:
    sodesol2, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, rb)
else:
    sodesol2, rb = None, None

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

