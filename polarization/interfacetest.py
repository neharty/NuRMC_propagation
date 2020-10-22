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
theta = np.pi/4
phi = 0

ninit = 1
kinit = get_dir(theta,phi)
sigma = np.array([0,0,1])

nd = np.array([1.2, 1.7, 1.6])
eps = np.diag(nd**2)

# optimization part (2 outgoing k vectors)

def f1(x):
    thet = x[0]
    t = x[2]
    k = get_dir(thet, phi)
    n1, n2 = get_n(k, nd)
    return np.linalg.norm(n1*k - ninit*kinit - t*sigma)

def f2(x):
    thet = x[0]
    t = x[2]
    k = get_dir(thet, phi)
    n1, n2 = get_n(k, nd)
    return np.linalg.norm(n2*k - ninit*kinit - t*sigma)

def fresnel(x):
    return check(get_dir(x[0], phi), x[1], nd)

def thetaconst(x):
    return x[0]

tc = LinearConstraint(thetaconst, -np.pi/2, np.pi/2)
nlc = NonlinearConstraint(fresnel, 0, 0)

n1, n2 = get_n(kinit, nd)

sol1 = minimize(f1, (theta, n1, 0), constraints=nlc)
sol2 = minimize(f2, (theta, n2, 0), constraints=nlc)

print(kinit)

ksol1 = sol1.x[1]*get_dir(sol1.x[0], phi)
k1dir = get_dir(sol1.x[0], phi)
print(k1dir, np.linalg.norm(ksol1))

ksol2 = sol2.x[1]*get_dir(sol2.x[0], phi)
k2dir = get_dir(sol2.x[0], phi)
print(k2dir)

# get polarizations
# TM wave initially
Einit = vector.vector(np.array([np.cos(theta), 0, -np.sin(theta)]))
n1 = np.linalg.norm(ksol1)
J = eps - n1**2*np.eye(3)
print(np.linalg.det(J))
adjJ = np.array([[J[1,1]*J[2,2],0,0],
    [0, J[0,0]*J[2,2], 0],
    [0, 0, J[0,0]*J[1,1]]]).T
print(adjJ)
ehat = adjJ.dot(k1dir)
d1 = eps.dot(ehat)/np.linalg.norm(eps.dot(ehat))
print(d1)
print(d1.dot(k1dir))
x = eps.dot(p.get_E(k1dir, n1, eps))
x = x/np.linalg.norm(x)
print(x)

