import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint

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

theta = np.random.random()*np.pi - np.pi/2
phi = np.random.random()*np.pi*2

ninit = 1
kinit = get_dir(theta, phi)
kplane = ninit*np.array([np.sin(theta), np.cos(theta)]) # for graphing
sigma = np.array([0,0,1])

nd = np.array([1.2, 1.7, 1.6])
nd2 = nd**2

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

fig, ax = plt.subplots()

# optimization part

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

ksol1 = sol1.x[1]*get_dir(sol1.x[0], phi)
ksol1plane = sol1.x[1]*np.array([np.sin(sol1.x[0]), np.cos(sol1.x[0])])

ksol2 = sol2.x[1]*get_dir(sol2.x[0], phi)
ksol2plane = sol2.x[1]*np.array([np.sin(sol2.x[0]), np.cos(sol2.x[0])])

ax.quiver(0,0, kplane[0], kplane[1], angles='xy', scale_units='xy', scale=1)
ax.quiver(0,0, ksol1plane[0], ksol1plane[1], angles='xy', scale_units='xy', scale=1, color='b')
ax.quiver(0,0, ksol2plane[0], ksol2plane[1], angles='xy', scale_units='xy', scale=1, color='g')
ax.axvline(ninit*np.sin(theta), linestyle='dashed', color='r')
ax.plot(r1, z1)
ax.plot(r2, z2)
ax.set_ylim([-0.1, max(max(z1), max(z2))+0.1])
plt.show()
