import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light
import snell_fns as sf

'''
parser = argparse.ArgumentParser(description='compare snells law ODE with just the regular one')
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

rmax = 1000
z0 = -300
zm = -200
dr = 10
dz = 10
print(sf.cont)
cont = sf.cont

def initialangle(zd, z0):
    if zd-z0 < 0:
        return np.pi/4
    if zd-z0 >= 0:
        return np.pi/2 - np.arctan((zd-z0)/rmax)

'''
#example for plotting

podesol1 = get_ray(objfn, podes, initialangle(zm, z0), rmax, z0, zm, dr, np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax))
podesol2 = get_ray(objfn, podes, np.pi/2-np.arctan((-zm-z0)/rmax), rmax, z0, zm, dr, np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))

sodesol1 = get_ray(objfn, sodes, initialangle(zm, z0), rmax, z0, zm, dr, np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax))
sodesol2 = get_ray(objfn, sodes, np.pi/2-np.arctan((-zm-z0)/rmax), rmax, z0, zm, dr, np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))

plt.plot(podesol1.t, podesol1.y[1], label = 'p1-wave')
plt.plot(podesol2.t, podesol2.y[1], '-.', label = 'p2-wave')
plt.plot(sodesol1.t, sodesol1.y[1], '--', label = 's1-wave')
plt.plot(sodesol2.t, sodesol2.y[1], '.', label = 's2-wave')
plt.plot(rmax, zm, 'D', label = 'antenna')
plt.plot(0, z0, '*', label = 'source')
plt.xlabel('r')
plt.ylabel('z')
plt.legend()
#plt.savefig('snells_uniaxial_example'+str(cont).replace('.','')+'.png', dpi=600)
plt.show()
plt.clf()

input()
'''
#sweeping data
datanum = 11
zarr = np.linspace(0, -2000, num=datanum)
tmptab = np.zeros((datanum, 6))
for j in range(len(zarr)):
    z0 = zarr[j]
    print('\ndepth: ', z0)
    
    plt.figure(1)
    #podesol1 = get_ray(objfn, podes, initialangle(zm, z0), rmax, z0, zm, dr, np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax))
    #podesol2 = get_ray(objfn, podes, initialangle(-zm, z0), rmax, z0, zm, dr, np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
    print('inital guesses: ', np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
    podesol1 = sf.get_ray(sf.objfn, sf.podes, initialangle(zm, z0), rmax, z0, zm, dr, np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax))
    podesol2 = sf.get_ray(sf.objfn, sf.podes, np.pi/2-np.arctan((-zm-z0)/rmax), rmax, z0, zm, dr, np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
    if(podesol1 is not None):
        tp1 = 1e9*podesol1.y[2,-1]/speed_of_light
        plt.plot(podesol1.t, podesol1.y[1], color='blue')
    if(podesol2 is not None):
        tp2 = 1e9*podesol2.y[2,-1]/speed_of_light
        plt.plot(podesol2.t, podesol2.y[1], '--', color='orange')
    
    plt.figure(2)
    #sodesol1 = get_ray(objfn, sodes, initialangle(zm, z0), rmax, z0, zm, dr, np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax))
    #sodesol2 = get_ray(objfn, sodes, initialangle(-zm, z0), rmax, z0, zm, dr, np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
    sodesol1 = sf.get_ray(sf.objfn, sf.sodes, initialangle(zm, z0), rmax, z0, zm, dr, np.arcsin(1/sf.ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax))
    sodesol2 = sf.get_ray(sf.objfn, sf.sodes, np.pi/2-np.arctan((-zm-z0)/rmax), rmax, z0, zm, dr, np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))

    if(sodesol1 is not None):
        ts1 = 1e9*sodesol1.y[2,-1]/speed_of_light
        plt.plot(sodesol1.t, sodesol1.y[1], '', color='orange')
    if(sodesol2 is not None):
        ts2 = 1e9*sodesol2.y[2,-1]/speed_of_light
        plt.plot(sodesol2.t, sodesol2.y[1], '--', color='c')

    #tmptab[j,0], tmptab[j,2], tmptab[j, 4] = pminsol.x[0], sminsol.x[0], pminsol.x[0]-sminsol.x[0]
    #tmptab[j,1], tmptab[j,3], tmptab[j, 5] = tp, ts, tp-ts

#plt.title('blues = p-waves, oranges = s-waves, cont = '+str(cont))
plt.title('s-waves, cont = '+str(cont))
plt.plot(rmax, zm, 'D', label = 'antenna')
plt.xlabel('r')
plt.ylabel('z')
plt.legend()

plt.figure(1)
plt.title('p-waves, cont = '+str(cont))
plt.plot(rmax, zm, 'D', label = 'antenna')
plt.xlabel('r')
plt.ylabel('z')
plt.legend()

#tab = pd.DataFrame(data=tmptab, index=zarr, columns=['p launch', 't_p [ns]', 's launch', 't_s [ns]', 'p-s delta launch', 'p-s delta t [ns]'])

#tab.to_csv('snells_uniaxial_data'+str(cont).replace('.','')+'.csv')

#plt.savefig('snell_uniaxial_sweep_'+str(cont).replace('.','')+'.png', dpi=600)
'''
fl = open('maxerr.txt', 'a')
fl.write(str(dl)+','+str(np.max(np.abs(sol.sol(r2)[1] - z2))))
fl.write('\n')
fl.close()
'''
plt.show()
