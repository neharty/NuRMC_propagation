import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
import vector
import polarizationfns as p

def get_n(s,n):
    sx, sy, sz = s[0], s[1], s[2]
    vx, vy, vz = 1/n[0], 1/n[1], 1/n[2]
    v1, v2 = np.sqrt(np.roots([sx**2+sy**2+sz**2, -sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2), (sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2]))
    return 1/v1, 1/v2

def check(s, n, narr):
    sx, sy, sz = s
    v = 1/n
    varr = 1/narr
    vx, vy, vz = varr
    return (sx**2+sy**2+sz**2)*v**4 + (-sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2))*v**2 + ((sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2)
    #return sx**2/(v**2 - vx**2) + sy**2/(v**2-vy**2) + sz**2/(v**2 - vz**2)

def get_dir(theta, phi):
    return np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

#initializing stuff
theta = np.random.random()*np.pi/2
phi = np.random.random()*2*np.pi

ndinit = np.array([1.2, 1.5, 1.8])
epsinit = np.diag(ndinit**2)
kinit = get_dir(theta,phi)
p1, p2 = p.get_null(kinit, epsinit)
print(p1.T[0])
print(p2.T[0])
n1i, n2i = get_n(kinit, ndinit)

kplane1 = n1i*np.array([np.sin(theta), np.cos(theta)]) # for graphing
kplane2 = n2i*np.array([np.sin(theta), np.cos(theta)])

#plane normal
sigma = np.array([0,0,1])

#2nd interface
nd = np.array([1.5, 1.7, 1.6])
eps = np.diag(nd**2)

# optimization part (2 outgoing k vectors)
def f1(x, n):
    thet = x[0]
    t = x[2]
    k = get_dir(thet, phi)
    n1, n2 = get_n(k, nd)
    return np.linalg.norm(n1*k - n*kinit - t*sigma)

def f2(x, n):
    thet = x[0]
    t = x[2]
    k = get_dir(thet, phi)
    n1, n2 = get_n(k, nd)
    return np.linalg.norm(n2*k - n*kinit - t*sigma)

def fresnel(x):
    return check(get_dir(x[0], phi), x[1], nd)

def thetaconst(x):
    return x[0]

tc = LinearConstraint(thetaconst, -np.pi/2, np.pi/2)
nlc = NonlinearConstraint(fresnel, 0, 0)

n1, n2 = get_n(kinit, nd)


sol11 = minimize(f1, (theta, n1, 0), args=n1i, constraints=nlc)
sol21 = minimize(f2, (theta, n2, 0), args=n1i, constraints=nlc)

sol12 = minimize(f1, (theta, n1, 0), args=n2i, constraints=nlc)
sol22 = minimize(f2, (theta, n2, 0), args=n2i, constraints=nlc)

ksol11 = sol11.x[1]*get_dir(sol11.x[0], phi)
#print(p.get_E(get_dir(sol11.x[0], phi), sol11.x[1], eps))
ksol11plane = sol11.x[1]*np.array([np.sin(sol11.x[0]), np.cos(sol11.x[0])])

ksol21 = sol21.x[1]*get_dir(sol21.x[0], phi)
#print(p.get_E(get_dir(sol21.x[0], phi), sol21.x[1], eps))
ksol21plane = sol21.x[1]*np.array([np.sin(sol21.x[0]), np.cos(sol21.x[0])])

ksol12 = sol12.x[1]*get_dir(sol12.x[0], phi)
#print(p.get_E(get_dir(sol12.x[0], phi), sol12.x[1], eps))
ksol12plane = sol12.x[1]*np.array([np.sin(sol12.x[0]), np.cos(sol12.x[0])])

ksol22 = sol22.x[1]*get_dir(sol22.x[0], phi)
#print(p.get_E(get_dir(sol22.x[0], phi), sol22.x[1], eps))
ksol22plane = sol22.x[1]*np.array([np.sin(sol22.x[0]), np.cos(sol22.x[0])])

num = 100
angs = np.linspace(-np.pi/2, np.pi/2, num=num)

r1 = np.zeros(num)
z1 = np.zeros(num)
r2 = np.zeros(num)
z2 = np.zeros(num)

for i in range(num):
    ang = angs[i]
    k = get_dir(ang, phi)
    n1, n2 = get_n(k, nd)

    r1[i] = n1*np.sin(ang)
    z1[i] = n1*np.cos(ang)

    r2[i] = n2*np.sin(ang)
    z2[i] = n2*np.cos(ang)

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)

ax1.quiver(0,0, kplane1[0], kplane1[1], angles='xy', scale_units='xy', scale=1)
ax1.quiver(0,0, ksol11plane[0], ksol11plane[1], angles='xy', scale_units='xy', scale=1, color='b')
ax1.quiver(0,0, ksol21plane[0], ksol21plane[1], angles='xy', scale_units='xy', scale=1, color='g')
ax1.axvline(n1i*np.sin(theta), linestyle='dashed', color='r')
ax1.plot(r1, z1)
ax1.plot(r2, z2)
ax1.set_xlabel('r')
ax1.set_ylabel('z')
ax1.set_ylim([-0.1, max(max(z1), max(z2))+0.1])

ax2.quiver(0,0, kplane2[0], kplane2[1], angles='xy', scale_units='xy', scale=1)
ax2.quiver(0,0, ksol12plane[0], ksol12plane[1], angles='xy', scale_units='xy', scale=1, color='b')
ax2.quiver(0,0, ksol22plane[0], ksol22plane[1], angles='xy', scale_units='xy', scale=1, color='g')
ax2.axvline(n2i*np.sin(theta), linestyle='dashed', color='r')
ax2.plot(r1, z1)
ax2.plot(r2, z2)
ax2.set_xlabel('r')
ax2.set_ylabel('z')
ax2.set_ylim([-0.1, max(max(z1), max(z2))+0.1])

#plt.title(r'$\phi_i$ = '+str(phi)+' '+r'$\theta_i = $' + str(theta))
#plt.savefig('interface_optimization.pdf')
plt.show()

# get polarizations
# TM wave initially

J = eps - sol11.x[1]**2*np.eye(3)
print(np.linalg.det(J))
adjJ = np.array([[J[1,1]*J[2,2],0,0],
    [0, J[0,0]*J[2,2], 0],
    [0, 0, J[0,0]*J[1,1]]]).T
ehat = adjJ.dot(sol11.x[1]*get_dir(sol11.x[0], phi))
print(ehat/np.linalg.norm(ehat))
d1 = eps.dot(ehat)/np.linalg.norm(eps.dot(ehat))
print(d1)
print(d1.dot(k1dir))
x = eps.dot(p.get_E(k1dir, n1, eps))
x = x/np.linalg.norm(x)
print(x)


