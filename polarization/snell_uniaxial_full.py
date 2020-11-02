import numpy as np
from scipy.integrate import solve_ivp
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import curve_fit
from tabulate import tabulate

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

'''
    # fit data for permittivity tensor
    epsdata = pd.read_csv('./epsdata/evals_vs_depth.csv')
    print(epsdata)
    zdepth = -np.array(epsdata['Nominal Depth'])
    e1 = 3.157 + 0.034*np.array(epsdata['E1'])
    e2 = 3.157 + 0.034*np.array(epsdata['E2'])
    e3 = 3.157 + 0.034*np.array(epsdata['E3'])

    e12 = (e1+e2)/2

    nxy = np.sqrt(e12)
    nz = np.sqrt(e3)

    del e1, e2, e3, e12, epsdata

    def nfit(z, a, b, c):
        z0 = z[0]
        return a + b*np.exp((z-z0)/c)

    nxyparams, junk = curve_fit(nfit, zdepth, nxy)
    nzparams, junk = curve_fit(nfit, zdepth, nz)

    plt.plot(zdepth, nxy, '.', label=r'$n_{o}$')
    plt.plot(zdepth, nz, '.', label=r'$n_{e}$')
    plt.plot(zdepth, nfit(zdepth, *nxyparams), label = r'$n_{o}$ fit')
    plt.plot(zdepth, nfit(zdepth, *nzparams), label = r'$n_{e}$ fit')
    plt.xlim([zdepth[0], zdepth[-1]])
    plt.legend()
    plt.xlabel('nominal depth [m]')
    plt.ylabel('index of refraction')
    plt.savefig('icemodel.png')
    plt.clf()

    zs = np.linspace(0, -2700, num=2701)
    plt.plot(zs, nfit(zs, *nxyparams), label='n_o')
    plt.plot(zs, nfit(zs, *nzparams), label='n_e')
    plt.show()
    plt.clf()
    '''
# ice parameters
nss = 1.35
nd = 1.78
c = 0.0132

d = 0.1

def nz(z):
    # z index of refraction function
    # from nice8 ARA model
    a = nd        
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if  z > d:
        return 1
    if np.abs(z) <= d:
        return (n2-n1)*z/(2*d) +(n2+n1)/2
    if z < -d:
        return a + b*np.exp(z*c) 

def no(z):
    # x-y plane index of refraction function
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if  z > d:
        return 1
    if np.abs(z) <= d:
        return (n2-n1)*z/(2*d) + (n2+n1)/2
    if z < -d:
        return 0.99*nz(z)

def eps(z):
    # epsilon is diagonal
    return np.diag([(no(z))**2, (no(z))**2, (nz(z))**2])

def dnodz(z):
    #derivative of x-y index of refraction
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if z > d:
        return 0
    if np.abs(z) <= d:
        return (n2-n1)/(2*d)
    if z < -d:
        return 0.99*dnzdz(z)

def dnzdz(z):
    #derivative of z index of refraction
    b = nss - nd
    return b*c*np.exp(z*c)

def ns(z):
    #s-polarization index of refraction
    return no(z)

def dnsdz(z):
    #derivative of s-polarization index of refraction
    return dnodz(z)

def npp(theta, z):
    #p-polarizatoin index of refraction
    return no(z)*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)

def dnpdtheta(theta, z):
    #partial of np w.r.t. theta
    if z >= -d:
        return 0
    if z < -d:
        return no(z)*nz(z)*np.sin(theta)*np.cos(theta)*(nz(z)**2-no(z)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def dnpdz(theta, z):
    #partial of np w.r.t. z
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if z > d:
        return 0
    if np.abs(z) <= d:
        return (n2-n1)/(2*d)
    if z < -d:
        return (no(z)**3*dnzdz(z)*np.sin(theta)**2+dnodz(z)*nz(z)**3*np.cos(theta)**2)/(nz(z)**2*np.cos(theta)**2+no(z)**2*np.sin(theta)**2)**1.5

def podes(t, y):
    # odes for p-polarization
    # form is [d(theta)/dr, dzdr]
    return [-np.cos(y[0])*dnpdz(y[0], y[1])/(npp(y[0],y[1])*np.cos(y[0])+dnpdtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0])]

def sodes(t,y):
    # odes for s-polarization
    # form is [d(theta)/dr, dzdr]
    return [-dnsdz(y[1])/(ns(y[1])), 1/np.tan(y[0])]

def pfn(theta, rmax, z0, zm, dr):
    sol = solve_ivp(podes, [0, rmax], [theta, z0], method='DOP853', max_step = dr)
    zsol = sol.y[1,-1]
    return np.abs(zsol - zm)

def sfn(theta, rmax, z0, zm, dr):
    sol = solve_ivp(sodes, [0, rmax], [theta, z0], method='DOP853', max_step = dr)
    zsol = sol.y[1,-1]
    return np.abs(zsol - zm)

rmax = 1000
z0 = -300
zm = -200
dr = 10
dz = 10

def initialangle(zd, z0):
    if zd-z0 < 0:
        return np.pi/4
    if zd-z0 >= 0:
        return np.pi/2 - np.arctan((zm-z0)/rmax)

#example for plotting
pminsol = minimize(pfn, (initialangle(zm, z0)), args=(rmax, z0, zm, dr), tol = 1e-5)
podesol = solve_ivp(podes, [0, rmax], [pminsol.x[0], z0], method='DOP853', max_step = dr)

sminsol = minimize(sfn, (initialangle(zm,z0)), args=(rmax, z0, zm, dr), tol=1e-5)
sodesol = solve_ivp(sodes, [0, rmax], [sminsol.x[0], z0], method='DOP853', max_step = dr)

plt.plot(podesol.t, podesol.y[1], color='blue', label = 'p-wave')
plt.plot(sodesol.t, sodesol.y[1], '--', color='orange', label = 's-wave')
plt.plot(rmax, zm, 'D', label = 'antenna')
plt.plot(0, z0, '*', label = 'source')
plt.xlabel('r')
plt.ylabel('z')
plt.legend()
plt.savefig('snells_uniaxial_example.png', dpi=600)
plt.clf()

#sweeping data
zarr = np.linspace(0, -2000, num=21)
tmptab = np.zeros((21, 6))
for j in range(len(zarr)):
    z0 = zarr[j]
    pminsol = minimize(pfn, (initialangle(zm, z0)), args=(rmax, z0, zm, dr), tol = 1e-5)
    print('pminsol', pminsol.success, pminsol.message)
    podesol = solve_ivp(podes, [0, rmax], [pminsol.x[0], z0], method='DOP853', max_step = dr)
    tp = (10/3)*np.trapz([npp(podesol.y[0,i], podesol.y[1,i])*np.abs(1/np.sin(podesol.y[0,i])) for i in range(len(podesol.t))],  podesol.t)
    
    sminsol = minimize(sfn, (initialangle(zm,z0)), args=(rmax, z0, zm, dr), tol=1e-5)
    print('sminsol', sminsol.success, pminsol.message)
    sodesol = solve_ivp(sodes, [0, rmax], [sminsol.x[0], z0], method='DOP853', max_step = dr)
    ts = (10/3)*np.trapz([npp(sodesol.y[0,i], sodesol.y[1,i])*np.abs(1/np.sin(sodesol.y[0,i])) for i in range(len(sodesol.t))],  sodesol.t)

    tmptab[j,0], tmptab[j,2], tmptab[j, 4] = pminsol.x[0], sminsol.x[0], pminsol.x[0]-sminsol.x[0]
    tmptab[j,1], tmptab[j,3], tmptab[j, 5] = tp, ts, tp-ts
    if np.abs(podesol.y[1,-1] - zm) <= 1e-4 or np.abs(sodesol.y[1,-1] - zm)<= 1e-4:
        plt.plot(podesol.t, podesol.y[1], color='blue')
        plt.plot(sodesol.t, sodesol.y[1], '--', color='orange')

plt.title('blue = p-wave, orange = s-wave')
plt.plot(rmax, zm, 'D', label = 'antenna')
plt.xlabel('r')
plt.ylabel('z')
plt.legend()

tab = pd.DataFrame(data=tmptab, index=zarr, columns=['p launch', 't_p [ns]', 's launch', 't_s [ns]', 'p-s delta launch', 'p-s delta t [ns]'])
tab.to_csv('snells_uniaxial_data.csv')

plt.savefig('snell_uniaxial_sweep.png', dpi=600)
'''
fl = open('maxerr.txt', 'a')
fl.write(str(dl)+','+str(np.max(np.abs(sol.sol(r2)[1] - z2))))
fl.write('\n')
fl.close()
'''
#plt.show()
