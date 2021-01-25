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
import h5py
import random
import multiprocessing as mp

import snell_prop_fns as sf
from derivs import derivs_biaxial_c as dv

from ray import ray

sf.cont = 0.99
'''
rmax = 1000
z0 = -1000
'''
zm = -200
dr = 10
dz = 10

fl = h5py.File("1e18_n1e4.hdf5",'r')

indx = random.sample(range(len(fl['xx'])), 10)

xx, yy, z0s = np.array(fl['xx']), np.array(fl['yy']), np.array(fl['zz'])
z0s = z0s[indx]

cont = sf.cont
rmaxs = np.sqrt(xx[indx]**2 + yy[indx]**2)
times = np.zeros(len(z0s))

events = sf.hit_top

outp = mp.Queue()

phi = np.random.random()*2*np.pi

def getray(raytype, initguess):
    return sf.get_ray_1guess(sf.objfn, dv.odefns, rmax, z0, zm, dr, raytype, initguess)

if __name__ == "__main__":
    for i in range(len(z0s)):
        
        rmax = rmaxs[i]
        z0 = z0s[i]

        grazang = np.arcsin(np.trace(np.sqrt(dv.eps(0)))/(np.trace(np.sqrt(dv.eps(z0)))))

        directang = np.arctan(rmax/(np.abs(z0)-np.abs(zm)))

        mirrorang = np.arctan(rmax/(np.abs(z0)+np.abs(zm)))

        guess1 = min([grazang, directang, mirrorang])

        guess2 = max([grazang, directang, mirrorang])

        #pool = mp.Pool(processes=4)
        
        p1ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 1, 'p1')
        p2ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 1, 'p2')
        s1ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 2, 's1')
        s2ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 2, 's2')
        
        pc1 = mp.Process(target=p1ray.comp_ray, args=(guess1,))
        pc2 = mp.Process(target=p2ray.comp_ray, args=(guess2,))
        sc1 = mp.Process(target=s1ray.comp_ray, args=(guess1,))
        sc2 = mp.Process(target=p2ray.comp_ray, args=(guess2,))

        ttmp = time.time()
        #p1, p2, s1, s2 = pool.starmap(getray, [(1, guess1), (1, guess2), (2, guess1), (2, guess2)])
        pc1.start(), pc2.start(), sc1.start(), sc2.start()
        pc1.join(), pc2.join(), sc1.join(), sc2.join()
        times[i] = time.time() - ttmp

print('approx max computation time:', max(times)/4)
print('avg time per ray:', np.average(times)/4)
