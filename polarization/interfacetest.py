import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def get_n(s,n):
    sx, sy, sz = s[0], s[1], s[2]
    vx, vy, vz = 1/n[0], 1/n[1], 1/n[2]
    v1, v2 = np.sqrt(np.roots([sx**2+sy**2+sz**2, -sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2), (sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2]))
    return 1/v1, 1/v2


'''
# optimization problem version
#test on x-y plane
phi=np.pi/4

sigma=np.array([np.cos(phi), np.sin(phi), 0])

s1 = np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0])

n1 = 1.2

narr = np.array([1.2, 1.5, 1.7])**2

#def fun (s):
#    n1p, n2p = get_n(s, narr)
#    return np.cross(sigma, n1*s1 - n1p*s)

sol = root(fun, s1)

print(sol.x-s1)

n1p, n2p = get_n(sol.x, narr)
print(np.linalg.norm(np.cross(sol.x, sigma))*n1p - n1*np.linalg.norm(np.cross(s1, sigma))) 
'''
'''
# analytic formulation

phi = np.random.random()*np.pi/2 # azimuth
theta = np.random.random()*np.pi/2 # zenith

narr = np.array([1.2, 1.5, 1.7])

s = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
n = 1.4
print(s)

phi = np.random.random()*np.pi/2 # azimuth
theta = np.random.random()*np.pi/2 # zenith

sigma = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
print(sigma)

alpha = -n*sigma.dot(s)
print(alpha)

np1, np2 = get_n(s+(alpha/n)*sigma, narr)
print(np1, np2)

sp1 = (n*s+alpha*sigma)/np1
sp2 = (n*s+alpha*sigma)/np2
print(sp1)
print(sp2)

print(np.linalg.norm(np.cross(sp1, sigma))*np1)
print(n*np.linalg.norm(np.cross(s, sigma)))

print(np.linalg.norm(np.cross(sp1, sigma))*np1 - n*np.linalg.norm(np.cross(s, sigma)))
print(np.linalg.norm(np.cross(sp2, sigma))*np2 - n*np.linalg.norm(np.cross(s, sigma)))
'''
def check_sol(s, n, nm):
    v2 = 1/n**2
    vx, vy, vz = 1/nm
    sx, sy, sz = s

    return np.abs((sx**2+sy**2+sz**2)*v2+(-sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2))*v2+(sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2)

def find_outgoing(s,n,sigma,narr,tol=1e-6):
    l = lambda t : n*s + t*sigma
    np1, np2, np3 = narr

    foundsol1 = False
    
    t0 = -n*s.dot(sigma)
    tmperr = check_sol(l(t0)/np.linalg.norm(l(t0)), np.linalg.norm(l(t0)), narr)
    dt = 1e-4
    
    while not foundsol1:
        t = t0 + dt
        sol1 = l(t)
        err = check_sol(sol1/np.linalg.norm(sol1), np.linalg.norm(sol1), narr)
        if (err <= tol):
            foundsol1 = True
            break
        '''
        if (tmperr < err):
            sol1 = l(t0)
            break
        '''
        t0 = t
        tmperr = err
    
    foundsol2 = False

    tmperr = check_sol(l(t0)/np.linalg.norm(l(t0)), np.linalg.norm(l(t0)), narr)
    
    print(err)
    
    while not foundsol2:
        t = t0 + dt
        sol2 = l(t)
        err = check_sol(sol2/np.linalg.norm(sol2), np.linalg.norm(sol2), narr)
        if (err <= tol):
            foundsol2 = True
            break
        t0 = t
    
    return sol1, sol2

narr = np.array([1.2, 1.5, 1.7])

sigma = np.array([0,0,1])

phi = np.random.random()*2*np.pi # azimuth
theta = np.random.random()*np.pi/2 # zenith

s = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
n = 1.4

print(find_outgoing(s,n,sigma,narr, tol=1e-3))
