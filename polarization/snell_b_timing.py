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
import derivs_biaxial as dv


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
ptimes = np.zeros(len(z0s))
stimes = np.zeros(len(z0s))
events = sf.hit_top

outp = mp.Queue()

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

        pool = mp.Pool(processes=4)
        
        ttmp = time.time()
        p1, p2, s1, s2 = pool.starmap(getray, [(1, guess1), (1, guess2), (2, guess1), (2, guess2)])
        #p2 = pool.apply(getray, args=(1, guess2))
        
        #s1 = pool.apply(getray, args=(2, guess1))
        #s2 = pool.apply(getray, args=(2, guess2))

        # from general quadratic formula
    
        stimes[i] = time.time() - ttmp

        #solp1, solp2, sols1, sols2 = outp.get(), outp.get(), outp.get(), outp.get()
        print(p1.y[1,-1])
        '''
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
        '''

print(max(ptimes), max(stimes))
print('avg p time:', np.average(ptimes))
print('avg s time per ray:', np.average(stimes)/4)
