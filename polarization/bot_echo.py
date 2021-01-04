import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light
#import snell_fns as sf
import snell_fns_x as sf
#import snell_fns_arclen_x as sf
import time
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
sf.cont = 1.001

rmax = 50
z0 = -1
zm = -1
dr = 10
dz = 10
print(sf.cont)
cont = sf.cont

#sweeping data
datanum = 11
#phiarr = np.linspace(0, np.pi/2, num=datanum)
phiarr = np.array([-np.pi/6, -np.pi/3, 0, np.pi/6, np.pi/3, np.pi/2])
tmptab = np.zeros((datanum, 6))

rguess = np.pi/2 + 0.01

for j in range(len(phiarr)):
    sf.phi = phiarr[j]
    print('\nphi: ', sf.phi)
    
    rb = 0

    ptime = time.time()
    podesol1, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, rguess)
    if rb is not None:
        podesol2, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, rb)
    else:
        podesol2, rb = None, None
    ptime = time.time() - ptime

    if(podesol1 is not None):
        tp1 = 1e9*podesol1.y[2,-1]/speed_of_light
        tmptab[j, 0]  = tp1
        plt.plot(podesol1.t, podesol1.y[1], label = 'p1')
    if(podesol2 is not None):
        tp2 = 1e9*podesol2.y[2,-1]/speed_of_light
        tmptab[j, 3] = tp2
        plt.plot(podesol2.t, podesol2.y[1], label='p2')
    
    stime = time.time()
    sodesol1, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, rguess)#, np.pi/2-np.arctan((-zm-z0)/rmax))
    if rb is not None:
        sodesol2, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, rb)#np.pi/2-np.arctan((-zm-z0)/rmax))#, np.pi/2-np.arctan((zm-z0)/rmax))
    else:
        sodesol2, rb = None, None
    stime = time.time() - stime

    if(sodesol1 is not None):
        ts1 = 1e9*sodesol1.y[2,-1]/speed_of_light
        tmptab[j, 1] = ts1
        plt.plot(sodesol1.t, sodesol1.y[1], label='s1')
    if(sodesol2 is not None):
        ts2 = 1e9*sodesol2.y[2,-1]/speed_of_light
        tmptab[j, 4] = ts2
        plt.plot(sodesol2.t, sodesol2.y[1], label='s2')

    if(podesol1 is not None and sodesol1 is not None):
        tmptab[j, 2] = tp1-ts1
    if(podesol2 is not None and sodesol2 is not None):
        tmptab[j, 5] = tp2-ts2
    
    print(ts1, tp1, ts1 - tp1)
    print(ts2, tp2, ts2 - tp2)
    
    
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(podesol1.t, podesol1.y[2])
    plt.plot(podesol2.t, podesol2.y[2])
    plt.plot(sodesol1.t, sodesol1.y[2])
    plt.plot(sodesol2.t, sodesol2.y[2])
    plt.show()
    plt.clf()
    

tab = pd.DataFrame(data=tmptab, index=phiarr, columns=['p1 travel time [ns]', 's1 travel time [ns]', 'p-s delta t 1 [ns]', 'p2 travel time [ns]', 's2 travel time [ns]', 'p-s delta t 2 [ns]'])
tab.to_csv('bot_echo_times_d'+str(np.abs(z0))+'_xb'+str(rmax)+'_c'+str(cont).replace('.','')+'.csv')

#plt.savefig('snell_uniaxial_sweep_'+str(cont).replace('.','')+'.png', dpi=600)
'''
fl = open('maxerr.txt', 'a')
fl.write(str(dl)+','+str(np.max(np.abs(sol.sol(r2)[1] - z2))))
fl.write('\n')
fl.close()
'''
#plt.show()
