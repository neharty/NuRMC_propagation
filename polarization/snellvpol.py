import numpy as np
from scipy.integrate import solve_ivp
import polarizationfns as pl
import matplotlib.pyplot as plt

theta0 = np.pi/4
z0 = 0

s = np.array([np.sin(theta0), 0, np.cos(theta0)])
s = s/np.linalg.norm(s)

L = 100

def nz(z):
    return 2-z/L

def eps(z):
    return np.diag([1.5**2, 1.6**2, (nz(z))**2])

n1, n2 = pl.get_n(s, np.diag(np.sqrt(eps(z0))))
p1, p2 = pl.get_null(s,eps(z0))

print(p1)
print(p2)

dl = 0.001
c1 = n1*np.sin(theta0)
c2 = n2*np.sin(theta0)

ent = int(2e2*1/dl)

z1 = np.zeros(ent)
r1 = np.zeros(ent)

z2 = np.zeros(ent)
r2 = np.zeros(ent)

z1[1] = z1[0] + dl*np.cos(theta0)
z2[1] = z2[0] + dl*np.cos(theta0)

r1[1] = r1[0] + dl*np.sin(theta0)
r2[1] = r2[0] + dl*np.sin(theta0)

theta1 = theta0
theta2 = theta0

up1 = True
up2 = True

#compute wave 1
for i in range(1, ent-1):
    n1s = np.sqrt(np.diag(eps(z1[i])))
    n1 = n1s[1]
    
    if c1/n1>1 and up1:
        theta1 = 2*np.pi-theta2
        up1 = False
    elif not up1:
        theta1 = 2*np.pi-np.arcsin(c1/n1)
    else:
        theta1 = np.arcsin(c1/n1)

    z1[i+1] = z1[i] + dl*np.cos(theta1)
    r1[i+1] = r1[i] + dl*np.sin(theta1)

#compute wave 2
for i in range(1, ent-1):
    n2s = np.sqrt(np.diag(eps(z2[i])))
    n2tmp = n2
    n2 = n2s[0]*n2s[2]/np.sqrt(n2s[0]**2*np.sin(theta2)**2 +n2s[2]**2*np.cos(theta2)**2)

    if c2/n2>1 and up2:
        theta2 = 2*np.pi-theta2
        up2 = False
        z2[i+1] = z2[i] - dl*np.cos(theta2)
        r2[i+1] = r2[i] - dl*np.sin(theta2)
    elif not up2:
        theta2 = 2*np.pi-np.arcsin(c2/n2)
        z2[i+1] = z2[i] - dl*np.cos(theta2)
        r2[i+1] = r2[i] - dl*np.sin(theta2)
    else:
        theta2 = np.arcsin(c2/n2)
        z2[i+1] = z2[i] + dl*np.cos(theta2)
        r2[i+1] = r2[i] + dl*np.sin(theta2)
    
ny = np.sqrt(eps(0)[0,0])

def nzp(z):
    return -1/L

def n(theta, z):
    return ny*nz(z)/np.sqrt(nz(z)**2*np.cos(theta)**2+ny**2*np.sin(theta)**2)

def dndtheta(theta, z):
    return ny*nz(z)*np.sin(theta)*np.cos(theta)*(nz(z)**2-ny**2)/(nz(z)**2*np.cos(theta)**2+ny**2*np.sin(theta)**2)**1.5

def dndz(theta, z):
    return ny**3*np.sin(theta)**2*nzp(z)/(nz(z)**2*np.cos(theta)**2+ny**2*np.sin(theta)**2)**1.5

def odes(t, y):
    # form is [d(theta)/dr, dzdr]
    return [-np.cos(y[0])*dndz(y[0], y[1])/(n(y[0],y[1])*np.cos(y[0])+dndtheta(y[0],y[1])*np.sin(y[0])), 1/np.tan(y[0])]

sol=solve_ivp(odes, [0, r2[-1]], [np.pi/4, 0], method='DOP853', max_step=dl, dense_output=True)

plt.plot(sol.t, sol.y[1], '--')
plt.xlabel('r')
plt.ylabel('z')

#plt.plot(r1, z1)
plt.plot(r2, z2)
#plt.show()

print(np.max(np.abs(sol.sol(r2)[1] - z2)))

