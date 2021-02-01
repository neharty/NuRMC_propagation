import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
import pandas as pd
from scipy.constants import speed_of_light
#import snell_fns as sf
import snell_prop_fns as sf
from derivs import derivs_xext as dv
import time
from multiprocessing import Queue, Process
from ray_old import ray
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
#sf.cont = args.cont
#sf.cont = 1.001

rmaxs = [2400, 3700, 3200, 3700]
z0s = [-1000, -1000, -1400, -1400]
zm = -200
dr = 10
dz = 10
#print(sf.cont)
#cont = sf.cont

#sweeping data
datanum = 11
phiarr = np.linspace(0, np.pi/2, num=datanum)
tmptab = np.zeros((datanum, 6))

def getray(raytype, initguess):
    return sf.get_ray_1guess(sf.objfn, dv.odefns, rmax, z0, zm, dr, raytype, initguess)

for k in range(len(rmaxs)):
    rmax = rmaxs[k]
    z0 = z0s[k]

    for j in range(len(phiarr)):
        dv.phi = phiarr[j]
        print('\nphi: ', dv.phi)
        phi = phiarr[j]

        grazang = np.arcsin(np.trace(np.sqrt(dv.eps(0)))/(np.trace(np.sqrt(dv.eps(z0)))))

        directang = np.arctan(rmax/(np.abs(z0)-np.abs(zm)))

        mirrorang = np.arctan(rmax/(np.abs(z0)+np.abs(zm)))

        guess1 = min([grazang, directang, mirrorang])

        guess2 = max([grazang, directang, mirrorang])

        p1ray = ray('derivs', 'derivs_xext', z0, zm, rmax, dr, phi, 1, 'p1')
        p2ray = ray('derivs', 'derivs_xext', z0, zm, rmax, dr, phi, 1, 'p2')
        s1ray = ray('derivs', 'derivs_xext', z0, zm, rmax, dr, phi, 2, 's1')
        s2ray = ray('derivs', 'derivs_xext', z0, zm, rmax, dr, phi, 2, 's2')

        pq1 = Queue()
        pq2 = Queue()
        sq1 = Queue()
        sq2 = Queue()

        p1p = Process(target=p1ray.comp_ray_parallel, args=(pq1, guess1))
        p2p = Process(target=p2ray.comp_ray_parallel, args=(pq2, guess2))
        s1p = Process(target=s1ray.comp_ray_parallel, args=(sq1, guess1))
        s2p = Process(target=s2ray.comp_ray_parallel, args=(sq2, guess2))

        p1p.start(), p2p.start(), s1p.start(), s2p.start()
        p1p.join(), p2p.join(), s1p.join(), s2p.join()

        p1ray.copy_ray(pq1.get()), p2ray.copy_ray(pq2.get()), s1ray.copy_ray(sq1.get()), s2ray.copy_ray(sq2.get())

        if(p1ray.odesol is not None):
            tmptab[j, 0]  = p1ray.travel_time
        if(p2ray.odesol is not None):
            tmptab[j, 3] = p2ray.travel_time

        if(s1ray.odesol is not None):
            tmptab[j, 1] = s1ray.travel_time
        if(s2ray.odesol is not None):
            tmptab[j, 4] = s2ray.travel_time

        if(p1ray.odesol is not None and s1ray.odesol is not None):
            tmptab[j, 2] = p1ray.travel_time - s1ray.travel_time
        if(p2ray.odesol is not None and s2ray.odesol is not None):
            tmptab[j, 5] = p2ray.travel_time - s2ray.travel_time
        
        print(p1ray.travel_time, s1ray.travel_time, p1ray.travel_time - s1ray.travel_time)
        print(p2ray.travel_time, s2ray.travel_time, p2ray.travel_time - s2ray.travel_time)
        
        tab = pd.DataFrame(data=tmptab, index=phiarr, columns=['p1 travel time [ns]', 's1 travel time [ns]', 'p-s delta t 1 [ns]', 'p2 travel time [ns]', 's2 travel time [ns]', 'p-s delta t 2 [ns]'])
        tab.to_csv('ARA_times_zcont1_d'+str(np.abs(z0))+'_xb'+str(rmax)+'.csv')

