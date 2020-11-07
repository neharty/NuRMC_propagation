import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light

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

#sim model parameters
d = 1
cont = 0.9

def nz(z):
    # z index of refraction function
    a = nd        
    b = nss - nd
    n1 = cont*(a + b*np.exp(-d*c))
    n2 = 1

    if  z > d:
        return 1
    if np.abs(z) <= d:
        return (n2-n1)*z/(2*d) +(n2+n1)/2
    if z < -d:
        return cont*no(z) 

def no(z):
    # x-y plane index of refraction function
    # from nice8 ARA model
    a = nd
    b = nss - nd
    n1 = a + b*np.exp(-d*c)
    n2 = 1

    if  z > d:
        return 1
    if np.abs(z) <= d:
        return (n2-n1)*z/(2*d) + (n2+n1)/2
    if z < -d:
        return a + b*np.exp(z*c)

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
        return b*c*np.exp(z*c)

def dnzdz(z):
    #derivative of z index of refraction
    a = nd
    b = nss - nd
    n1 = cont*(a + b*np.exp(-d*c))
    n2 = 1
    
    if z > d:
        return 0
    if np.abs(z) <= d:
        return (n2-n1)/(2*d)
    if z < -d:
        return cont*dnodz(z)

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
    return [-np.cos(y[0])*dnpdz(y[0], y[1])/(npp(y[0],y[1])*np.cos(y[0])+dnpdtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0]), npp(y[0], y[1])/np.abs(np.sin(y[0]))]

def sodes(t,y):
    # odes for s-polarization
    # form is [d(theta)/dr, dzdr]
    return [-dnsdz(y[1])/(ns(y[1])), 1/np.tan(y[0]), ns(y[1])/np.abs(np.sin(y[0]))]

def pfn(theta, rmax, z0, zm, dr):
    sol = solve_ivp(podes, [0, rmax], [theta, z0, 0], method='DOP853', max_step = dr)
    zsol = sol.y[1,-1]
    return zsol - zm

def sfn(theta, rmax, z0, zm, dr):
    sol = solve_ivp(sodes, [0, rmax], [theta, z0, 0], method='DOP853', max_step = dr)
    zsol = sol.y[1,-1]
    return zsol - zm

rmax = 1000
z0 = -600
zm = -200
dr = 10
dz = 10

def initialangle(zd, z0):
    if zd-z0 < 0:
        return np.pi/4
    if zd-z0 >= 0:
        return np.pi/2 - np.arctan((zd-z0)/rmax)

def get_ray(minfn, odefn, mininit, rmax, z0, zm, dr):
    minsol = root_scalar(minfn, x0=mininit, args=(rmax, z0, zm, dr), method='brentq', bracket=[mininit + 0.3, mininit-0.3])
    odesol = solve_ivp(odefn, [0, rmax], [minsol.root, z0, 0], method='DOP853', max_step = dr)
    return odesol

#example for plotting
'''
podesol1 = get_ray(pfn, podes, initialangle(zm, z0), rmax, z0, zm, dr)
podesol2 = get_ray(pfn, podes, np.pi/2-np.arctan((-zm-z0)/rmax), rmax, z0, zm, dr)

sodesol1 = get_ray(sfn, sodes, initialangle(zm, z0), rmax, z0, zm, dr)
sodesol2 = get_ray(sfn, sodes, np.pi/2-np.arctan((-zm-z0)/rmax), rmax, z0, zm, dr)

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

#sweeping data
datanum = 10
zarr = np.linspace(0, -2000, num=datanum)
tmptab = np.zeros((datanum, 6))
for j in range(len(zarr)):
    z0 = zarr[j]
    
    podesol1 = get_ray(pfn, podes, initialangle(zm, z0), rmax, z0, zm, dr)
    podesol2 = get_ray(pfn, podes, initialangle(-zm, z0), rmax, z0, zm, dr)
    tp1 = 1e9*podesol1.y[2,-1]/speed_of_light
    tp2 = 1e9*podesol2.y[2,-1]/speed_of_light
    
    sodesol1 = get_ray(sfn, sodes, initialangle(zm, z0), rmax, z0, zm, dr)
    sodesol2 = get_ray(sfn, sodes, initialangle(-zm, z0), rmax, z0, zm, dr)
    ts1 = 1e9*sodesol1.y[2,-1]/speed_of_light
    ts2 = 1e9*sodesol2.y[2,-1]/speed_of_light

    #tmptab[j,0], tmptab[j,2], tmptab[j, 4] = pminsol.x[0], sminsol.x[0], pminsol.x[0]-sminsol.x[0]
    #tmptab[j,1], tmptab[j,3], tmptab[j, 5] = tp, ts, tp-ts
    if np.abs(podesol1.y[1,-1] - zm) <= 1e-4 or np.abs(sodesol1.y[1,-1] - zm)<= 1e-4:
        plt.figure(1)
        plt.plot(podesol1.t, podesol1.y[1], color='blue')
        plt.plot(podesol2.t, podesol2.y[1], '--', color='r')
        
        plt.figure(2)
        plt.plot(sodesol1.t, sodesol1.y[1], '', color='orange')
        plt.plot(sodesol2.t, sodesol2.y[1], '--', color='c')

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
'''

#tab = pd.DataFrame(data=tmptab, index=zarr, columns=['p launch', 't_p [ns]', 's launch', 't_s [ns]', 'p-s delta launch', 'p-s delta t [ns]'])

#tab.to_csv('snells_uniaxial_data'+str(cont).replace('.','')+'.csv')

#plt.savefig('snell_uniaxial_sweep_'+str(cont).replace('.','')+'.png', dpi=600)
'''
fl = open('maxerr.txt', 'a')
fl.write(str(dl)+','+str(np.max(np.abs(sol.sol(r2)[1] - z2))))
fl.write('\n')
fl.close()
'''
#plt.show()

angs = np.linspace(np.arcsin(1/ns(z0))-0.1, np.pi/2, num=1000)
depths = np.linspace(-100, -2000, num=5)
zp = np.zeros(len(angs))
zs = np.zeros(len(angs))

for i in range(len(angs)):
    #for j in range(len(depths)):
        podesol = solve_ivp(podes, [0, rmax], [angs[i], z0, 0], method='LSODA', max_step=dr)
        sodesol = solve_ivp(sodes, [0, rmax], [angs[i], z0, 0], method='LSODA', max_step=dr)
        #plt.figure(1)
        #plt.plot(podesol.t, podesol.y[1])
        #plt.plot
    
        zp[i], zs[i] = podesol.y[1, -1], sodesol.y[1,-1]
        
#plt.figure(2)
plt.plot(angs, zp-zm,'-', label='p-waves')
plt.plot(angs, zs-zm, '--', label='s-waves')
plt.hlines(0, angs[0], angs[-1], colors='r', linestyles='-')
plt.vlines([np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax)], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='k')
plt.xlabel('initial ' + r'$\theta$')
plt.ylabel('antenna depth - final depth')
plt.title('radial distance = 1 km, initial depth = '+str(z0)+' m')
plt.savefig(str(z0)+'m_depthsvtheta.pdf')
#plt.show()

def get_bounds(leftguess, rightguess, odefn, rmax, z0, zm, dr):

    if(rightguess <= leftguess):
        print('invalid inputs: must have rightguess > leftguess')
        exit()
    
    xtol=1e-4
    maxiter=200
    
    dxi=1e-2
    dx = dxi
    zend1, zend2 = solve_ivp(odefn, [0, rmax], [leftguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm, solve_ivp(odefn, [0, rmax], [rightguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm
    while np.sign(zend1) == np.sign(zend2) and rightguess <= np.pi/2:
        leftguess += dx
        rightguess += dx
        zend1, zend2 = solve_ivp(odefn, [0, rmax], [leftguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm, solve_ivp(odefn, [0, rmax], [rightguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm
    #print('z ends:', np.sign(zend1), np.sign(zend2))
    print('adjusted interval:', leftguess, rightguess)
    if rightguess > np.pi/2:
        print('ERROR: no interval found')
        return None, None
        #plt.vlines([np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax)], min(min(zp), min(zs))-10, max(max(zp), max(zs)))
    else:
        #tighten left bound
        inum = 0
        lastguess = leftguess
        nextguess = leftguess
        while(dx >= xtol and inum < maxiter):
            inum += 1
            nextguess = lastguess + dx
            #print('-----\nstart', lastguess, nextguess, dx, inum)
            #print(np.sign(solve_ivp(odefn, [0, rmax], [nextguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm), np.sign(solve_ivp(odefn, [0, rmax], [rightguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm))
            if (np.sign(solve_ivp(odefn, [0, rmax], [nextguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm) != np.sign(solve_ivp(odefn, [0, rmax], [rightguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm)):
                lastguess = nextguess
                #print('1st if', lastguess, nextguess, dx)
            else:
                #nextguess = lastguess
                #dx = dx/2
                #print('else', lastguess, nextguess, dx)
                break

        print('lb dx, iters:', dx, inum)
        lb = lastguess

        #tighten right bound
        dx = dxi
        inum = 0
        lastguess = rightguess
        nextguess = rightguess
        while(dx >= xtol and inum < maxiter):
            inum += 1
            nextguess = lastguess-dx
            #print('-----\nstart', lastguess, nextguess, dx, inum)
            #print(np.sign(solve_ivp(odefn, [0, rmax], [nextguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm), np.sign(solve_ivp(odefn, [0, rmax], [lb, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm))
            if np.sign(solve_ivp(odefn, [0, rmax], [nextguess, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm) != np.sign(solve_ivp(odefn, [0, rmax], [lb, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm):
                lastguess = nextguess
            else:
                #nextguess = lastguess
                #dx = dx/2
                break

        print('rb dx, iters:', dx, inum)
        rb = lastguess

    print('returned bounds:', lb, rb)
    print('signs:', np.sign(solve_ivp(odefn, [0, rmax], [lb, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm), np.sign(solve_ivp(odefn, [0, rmax], [rb, z0, 0], method='DOP853', max_step=dr).y[1,-1]-zm))
    return lb, rb

print('p1 bounds')
print('initals:', np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax))
p1lb, p1rb = get_bounds(np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), podes, rmax, z0, zm, dr) 
print('p2 bounds')
p2lb, p2rb = get_bounds(np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax), podes, rmax, z0, zm, dr)
if p1lb is not None:
    plt.vlines(p1lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='orange')
    plt.vlines(p1rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')
    plt.vlines(p2lb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='orange')
    plt.vlines(p2rb, min(min(zp), min(zs)), max(max(zp), max(zs)), colors='g')
    plt.show()
else:
    print('AAAHHHHHHHHHH')
    input()

print('p2 bounds')
#p2lb, p2rb = get_bounds(np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax), podes)
#s1bs, s2bs = get_bounds(np.arcsin(1/ns(z0)), np.pi/2-np.arctan((-zm-z0)/rmax), podes), get_bounds(np.pi/2-np.arctan((-zm-z0)/rmax), np.pi/2-np.arctan((zm-z0)/rmax), sodes)

if p1lb is not None and p2lb is not None:
    plt.vlines([p1lb, p1ub, p2lb, p2ub], min(min(zp), min(zs)), max(max(zp), max(zs)), colors='k')
else:
    print('uh oh (press ctrl+c to exit)')
    input()

plt.xlabel('initial ' + r'$\theta$')
plt.ylabel('antenna depth - final depth')
plt.title('radial distance = 1 km, initial depth = '+str(z0)+' m')
plt.show()
