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

def pguess(z0):
    if np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1))+0.01 < np.pi/2-np.arctan((-zm-z0)/rmax):
        return np.arctan(sf.cont/np.sqrt(sf.cont**2*sf.ns(z0)**2 - 1))+0.01
    else: 
        return np.pi/2-np.arctan((-zm-z0)/rmax)

def sguess(z0):
    if np.arcsin(1/sf.ns(z0))+0.01 < np.pi/2-np.arctan((-zm-z0)/rmax):
        return np.arcsin(1/sf.ns(z0))+0.01
    else: 
        return np.pi/2-np.arctan((-zm-z0)/rmax)

#example for plotting
    
podesol1, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, pguess(z0))#, np.pi/2-np.arctan((-zm-z0)/rmax))
if rb is not None:
    podesol2, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, rb)#np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
else:
    podesol2, rb = None, None

sodesol1, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, sguess(z0))#, np.pi/2-np.arctan((-zm-z0)/rmax))
if rb is not None:
    sodesol2, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, rb)#np.pi/2-np.arctan((-zm-z0)/rmax))#, np.pi/2-np.arctan((zm-z0)/rmax))
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
plt.savefig('snells_uniaxial_example'+str(cont).replace('.','')+'.png', dpi=600)
#plt.show()
plt.clf()

#input()

#sweeping data
datanum = 11
zarr = np.linspace(0, -2000, num=datanum)
#tab = pd.DataFrame(index=zarr, columns=['p1 launch', 'p1 travel time [ns]', 's1 launch', 's1 travel time [ns]', 'p-s delta launch 1', 'p-s delta t 1 [ns]', 'p2 launch', 'p2 travel time [ns]', 's2 launch', 's2 travel time [ns]', 'p-s delta launch 2', 'p-s delta t 2 [ns]'])
tmptab = np.zeros((datanum, 16))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for j in range(len(zarr)):
    z0 = zarr[j]
    print('\ndepth: ', z0)
    
    print('inital guesses: ', np.arcsin(1/sf.nstmp(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
    rb = 0
    podesol1, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, pguess(z0))#, np.pi/2-np.arctan((-zm-z0)/rmax))
    if rb is not None:
        podesol2, rb = sf.get_ray_1guess(sf.objfn, sf.podes, rmax, z0, zm, dr, rb)#np.pi/2-np.arctan((-zm-z0)/rmax))#, np.pi/2-np.arctan((zm-z0)/rmax))
    else:
        podesol2, rb = None, None

    if(podesol1 is not None):
        tp1 = 1e9*podesol1.y[2,-1]/speed_of_light
        ax1.plot(podesol1.t, podesol1.y[1], 'b-', label=str(z0))
        tmptab[j, 0] , tmptab[j, 1], tmptab[j,2] = podesol1.y[0,0], tp1, np.abs(np.abs(sf.npp(podesol1.y[0,0], z0)*np.sin(podesol1.y[0,0])) - np.abs(sf.npp(podesol1.y[0, -1], podesol1.y[1, -1])*np.sin(podesol1.y[0, -1])))
    if(podesol2 is not None):
        tp2 = 1e9*podesol2.y[2,-1]/speed_of_light
        ax1.plot(podesol2.t, podesol2.y[1], '--', color='orange', label=str(z0))
        tmptab[j, 8] , tmptab[j, 9], tmptab[j,10] = podesol2.y[0,0], tp2, np.abs(np.abs(sf.npp(podesol1.y[0,0], z0)*np.sin(podesol1.y[0,0])) - np.abs(sf.npp(podesol1.y[0, -1], podesol1.y[1, -1])*np.sin(podesol1.y[0, -1])))
    
    sodesol1, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, sguess(z0))#, np.pi/2-np.arctan((-zm-z0)/rmax))
    if rb is not None:
        sodesol2, rb = sf.get_ray_1guess(sf.objfn, sf.sodes, rmax, z0, zm, dr, rb)#np.pi/2-np.arctan((-zm-z0)/rmax))#, np.pi/2-np.arctan((zm-z0)/rmax))
    else:
        sodesol2, rb = None, None

    if(sodesol1 is not None):
        ts1 = 1e9*sodesol1.y[2,-1]/speed_of_light
        ax2.plot(sodesol1.t, sodesol1.y[1], 'b-', label = str(z0))
        tmptab[j, 3] , tmptab[j, 4], tmptab[j,5] = sodesol1.y[0,0], ts1, np.abs(np.abs(sf.no(z0)*np.sin(sodesol1.y[0,0])) - np.abs(sf.no(sodesol1.y[1, -1])*np.sin(sodesol1.y[0, -1])))
    if(sodesol2 is not None):
        ts2 = 1e9*sodesol2.y[2,-1]/speed_of_light
        ax2.plot(sodesol2.t, sodesol2.y[1], '--', color='orange', label = str(z0))
        tmptab[j, 11] , tmptab[j, 12], tmptab[j,13] = sodesol2.y[0,0], ts2, np.abs(np.abs(sf.no(z0)*np.sin(sodesol1.y[0,0])) - np.abs(sf.no(sodesol1.y[1, -1])*np.sin(sodesol1.y[0, -1])))

    if(podesol1 is not None and sodesol1 is not None):
        tmptab[j, 6], tmptab[j,7] = podesol1.y[0,0] - sodesol1.y[0,0], tp1-ts1
    if(podesol2 is not None and sodesol2 is not None):
        tmptab[j, 14], tmptab[j, 15] = podesol2.y[0,0] - sodesol2.y[0,0], tp2-ts2

#plt.title('blues = p-waves, oranges = s-waves, cont = '+str(cont))
ax2.set_title('s-waves, distance = '+str(rmax) +' cont = '+str(cont))
ax2.plot(rmax, zm, 'D', label = 'antenna')
ax2.set_xlabel('r')
ax2.set_ylabel('z')
fig2.savefig('swavesweep'+str(cont).replace('.','')+'.png', dpi=600)
plt.close(fig2)
#plt.legend()

ax1.set_title('p-waves, distance = '+str(rmax) +' cont = '+str(cont))
ax1.plot(rmax, zm, 'D', label = 'antenna')
ax1.set_xlabel('r')
ax1.set_ylabel('z')
fig1.savefig('pwavesweep'+str(cont).replace('.','')+'.png', dpi=600)
#plt.legend()

#tab = pd.DataFrame(data=tmptab, index=zarr, columns=['p launch', 't_p [ns]', 's launch', 't_s [ns]', 'p-s delta launch', 'p-s delta t [ns]'])

tab = pd.DataFrame(data=tmptab, index=zarr, columns=['p1 launch', 'p1 travel time [ns]', 'p1 d(n sin(angle))', 's1 launch', 's1 travel time [ns]', 's1 d(n sin(angle))', 'p-s delta launch 1', 'p-s delta t 1 [ns]', 'p2 launch', 'p2 travel time [ns]', 'p2 d(n sin(angle))', 's2 launch', 's2 travel time [ns]', 's2 d(n sin(angle))', 'p-s delta launch 2', 'p-s delta t 2 [ns]'])
tab.to_csv('snells_uniaxial_data'+str(cont).replace('.','')+'.csv')

#plt.savefig('snell_uniaxial_sweep_'+str(cont).replace('.','')+'.png', dpi=600)
'''
fl = open('maxerr.txt', 'a')
fl.write(str(dl)+','+str(np.max(np.abs(sol.sol(r2)[1] - z2))))
fl.write('\n')
fl.close()
'''
#plt.show()
