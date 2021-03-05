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
import time
from ray_hamiltonian import rays
from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.utilities import medium


# calculate the contrast function

fl = pd.read_csv('epsdata/evals_vs_depth.csv')
depth = -np.array(fl['Nominal Depth'])

eperp = 3.157
deltae = 0.034

#puts eigenvals in same coords as jordan et al.
n3, n2, n1 = np.sqrt(eperp + np.array(fl['E1'])*deltae), np.sqrt(eperp + np.array(fl['E2'])*deltae), np.sqrt(eperp + np.array(fl['E3'])*deltae)

n2n3avg = (n2+n3)/2
z0 = depth[0]

def test_func(z, a, b, c):
    return b/(1+a*np.exp(c*(z-z0)))

p0 = [3, 0.0045, 3e-2]
params1, p = curve_fit(test_func, depth, (n2n3avg - n1)/(n2n3avg+n1), p0=p0)
print(params1)

cont = lambda zz: 1-test_func(zz, *params1)

nd = 1.78
delta_n = 0.43
z00 = 75.75757575757576
ice = lambda zz: nd - delta_n*np.exp(zz/z00)

print(cont(-2800)*ice(-2800))

def eps(pos):
    z = pos[2]
    tmp = np.array([[cont(z), 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
    return (ice(z)*tmp)**2


testdepths = np.linspace(-100, -1800, num=280)
#rmaxs = [2400, 3700, 3200, 3700]
#z0s = [-1000, -1000, -1400, -1400]
rmaxs = [3200]
z0s = [-1400]
zm = -200
dr = 10
dz = 10

datanum = 11
phiarr = np.linspace(0, np.pi/2, num=datanum)
tmptab = np.zeros((datanum, 6))

def getray(raytype, initguess):
    return sf.get_ray_1guess(sf.objfn, dv.odefns, rmax, z0, zm, dr, raytype, initguess)

for k in range(len(rmaxs)):
    rmax = rmaxs[k]
    z0 = z0s[k]

    for j in range(len(phiarr)):
        phi = phiarr[j]

        startpos = (0,0,z0)
        finishpos = (np.cos(phi)*rmax, np.sin(phi)*rmax, zm)

        rayz = rays(*startpos, *finishpos, eps)

        tmptab[j, 0] = rayz.get_time(0,0)
        tmptab[j, 3] = rayz.get_time(1,0)

        tmptab[j, 1] = rayz.get_time(0,1)
        tmptab[j, 4] = rayz.get_time(1,1)

        tmptab[j, 2] = rayz.get_time(0,0) - rayz.get_time(1,0)
        tmptab[j, 5] = rayz.get_time(0,1) - rayz.get_time(1,1)
        
        g = analyticraytracing.ray_tracing(startpos, finishpos, medium.get_ice_model('ARAsim_southpole'), n_frequencies_integration = 1)
        g.find_solutions()
        nrmc1 = g.get_path(0, n_points=100)
        nrmc2 = g.get_path(1, n_points=100)
        
        r11 = rayz.get_ray(0,0)
        r12 = rayz.get_ray(0,1)
        r21 = rayz.get_ray(1,0)
        r22 = rayz.get_ray(1,1)
        
        plt.plot(np.sqrt(r11.y[0,:]**2 + r11.y[1,:]**2), r11.y[2,:], '--.', label = 'p1')
        plt.plot(np.sqrt(r12.y[0,:]**2 + r12.y[1,:]**2), r12.y[2,:], '--*', label = 'p2')

        plt.plot(np.sqrt(r21.y[0,:]**2 + r21.y[1,:]**2), r21.y[2,:], label = 's1')
        plt.plot(np.sqrt(r22.y[0,:]**2 + r22.y[1,:]**2), r22.y[2,:], label = 's2')
        
        plt.plot(np.sqrt(nrmc1[:, 0]**2 + nrmc1[:, 1]**2), nrmc1[:,2], '--', label = 'a 1')
        plt.plot(np.sqrt(nrmc2[:, 0]**2 + nrmc2[:, 1]**2), nrmc2[:,2], '--', label = 'a 2')

        plt.legend()
        plt.show()
        plt.clf()
        
        print('\n', phi)
        print(rayz.get_time(0,0), rayz.get_time(1,0), rayz.get_time(0,0) - rayz.get_time(1,0))
        print(rayz.get_time(0,1), rayz.get_time(1,1), rayz.get_time(0,1) - rayz.get_time(1,1))
        print('\n')
        print(rayz.get_initial_E_pol(0,0), rayz.get_initial_E_pol(1,0), rayz.get_final_E_pol(0,0), rayz.get_final_E_pol(1,0))
        print(rayz.get_initial_E_pol(0,1), rayz.get_initial_E_pol(1,1), rayz.get_final_E_pol(0,1), rayz.get_final_E_pol(1,1))
        print('\n')

        tab = pd.DataFrame(data=tmptab, index=phiarr, columns=['p1 travel time [ns]', 's1 travel time [ns]', 'p-s delta t 1 [ns]', 'p2 travel time [ns]', 's2 travel time [ns]', 'p-s delta t 2 [ns]'])
        tab.to_csv('ARA_times_zcont1_d'+str(np.abs(z0))+'_xb'+str(rmax)+'.csv')

