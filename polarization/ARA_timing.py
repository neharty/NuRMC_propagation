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
import multiprocessing as mp
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

    grazang = np.arcsin(np.trace(np.sqrt(dv.eps(0)))/(np.trace(np.sqrt(dv.eps(z0)))))
    #dzg = sf.objfn(grazang, ode, rmax, z0, zm, dr)

    directang = np.arctan(rmax/(np.abs(z0)-np.abs(zm)))
    #dzd = sf.objfn(directang, ode, rmax, z0, zm, dr)

    mirrorang = np.arctan(rmax/(np.abs(z0)+np.abs(zm)))
    #dzm = sf.objfn(mirrorang, ode, rmax, z0, zm, dr)

    #sortangs = np.sort(np.abs(np.array([grazang, directang, mirrorang]) - zm))

    guess1 = min([grazang, directang, mirrorang])

    guess2 = max([grazang, directang, mirrorang])

    for j in range(len(phiarr)):
        dv.phi = phiarr[j]
        print('\nphi: ', dv.phi)
        
        pool = mp.Pool(processes=4)
        podesol1, podesol2, sodesol1, sodesol2 = pool.starmap(getray, [(1, guess1), (1, guess2), (2, guess1), (2, guess2)])
        if(podesol1 is not None):
            tp1 = 1e9*podesol1.y[2,-1]/speed_of_light
            tmptab[j, 0]  = tp1
        if(podesol2 is not None):
            tp2 = 1e9*podesol2.y[2,-1]/speed_of_light
            tmptab[j, 3] = tp2

        if(sodesol1 is not None):
            ts1 = 1e9*sodesol1.y[2,-1]/speed_of_light
            tmptab[j, 1] = ts1
        if(sodesol2 is not None):
            ts2 = 1e9*sodesol2.y[2,-1]/speed_of_light
            tmptab[j, 4] = ts2

        if(podesol1 is not None and sodesol1 is not None):
            tmptab[j, 2] = tp1-ts1
        if(podesol2 is not None and sodesol2 is not None):
            tmptab[j, 5] = tp2-ts2
        
        print(ts1, tp1, ts1 - tp1)
        print(ts2, tp2, ts2 - tp2)
        
        tab = pd.DataFrame(data=tmptab, index=phiarr, columns=['p1 travel time [ns]', 's1 travel time [ns]', 'p-s delta t 1 [ns]', 'p2 travel time [ns]', 's2 travel time [ns]', 'p-s delta t 2 [ns]'])
        tab.to_csv('ARA_times_zcont1_d'+str(np.abs(z0))+'_xb'+str(rmax)+'.csv')

