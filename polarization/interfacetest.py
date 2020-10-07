import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def get_n(s,n):
    sx, sy, sz = s[0], s[1], s[2]
    vx, vy, vz = 1/n[0], 1/n[1], 1/n[2]
    v1, v2 = np.sqrt(np.roots([sx**2+sy**2+sz**2, -sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2), (sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2]))
    return 1/v1, 1/v2

#test on x-y plane
phi=np.pi/4

sigma=np.array([np.cos(phi), np.sin(phi), 0])

s1 = np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0])

n1 = 1.2

narr = np.array([1.2, 1.5, 1.7])**2

def fun (s):
    n1p, n2p = get_n(s, narr)
    return np.cross(sigma, n1*s1 - n1p*s)

sol = root(fun, s1)

print(sol.x-s1)

n1p, n2p = get_n(sol.x, narr)
print(np.linalg.norm(np.cross(sol.x, sigma))*n1p - n1*np.linalg.norm(np.cross(s1, sigma))) 


