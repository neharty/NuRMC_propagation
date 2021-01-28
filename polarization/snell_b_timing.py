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
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Process, Queue

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

phi = np.random.random()*2*np.pi

if __name__ == "__main__":
    for i in range(len(z0s)):
        
        rmax = rmaxs[i]
        z0 = z0s[i]

        grazang = np.arcsin(np.trace(np.sqrt(dv.eps(0)))/(np.trace(np.sqrt(dv.eps(z0)))))

        directang = np.arctan(rmax/(np.abs(z0)-np.abs(zm)))

        mirrorang = np.arctan(rmax/(np.abs(z0)+np.abs(zm)))

        guess1 = min([grazang, directang, mirrorang])

        guess2 = max([grazang, directang, mirrorang])
        
        p1ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 1, 'p1')
        p2ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 1, 'p2')
        s1ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 2, 's1')
        s2ray = ray('derivs', 'derivs_biaxial_c', z0, zm, rmax, dr, phi, 2, 's2')
                
        pq1 = Queue()
        pq2 = Queue()
        sq1 = Queue()
        sq2 = Queue()
        
        p1p = Process(target=p1ray.comp_ray_parallel, args=(pq1, guess1))
        p2p = Process(target=p1ray.comp_ray_parallel, args=(pq2, guess2))
        s1p = Process(target=p1ray.comp_ray_parallel, args=(sq1, guess1))
        s2p = Process(target=p1ray.comp_ray_parallel, args=(sq2, guess2))
        
        p1p.start(), p2p.start(), s1p.start(), s2p.start()
        p1p.join(), p2p.join(), s1p.join(), s2p.join()

        p1ray.copy_ray(pq1.get()), p2ray.copy_ray(pq2.get()), s1ray.copy_ray(sq1.get()), s2ray.copy_ray(sq2.get())

        print('testprint:', p1ray.travel_time)
        plt.plot(p1ray.get_ray_r()[:], p1ray.get_ray_z()[:])
        plt.plot(p2ray.get_ray_r()[:], p2ray.get_ray_z()[:])
        plt.plot(s1ray.get_ray_r()[:], s1ray.get_ray_z()[:], '--')
        plt.plot(s2ray.get_ray_r()[:], s2ray.get_ray_z()[:], '--')
        plt.show()
        plt.clf()

        #times[i] = time.time() - ttmp

print('approx max computation time:', max(times)/4)
print('avg time per ray:', np.average(times)/4)
