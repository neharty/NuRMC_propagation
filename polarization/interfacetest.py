import numpy as np
import matplotlib.pyplot as plt

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

theta = np.pi/4
phi = np.pi/4

ninit = 1.5
kinit = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
sigma = np.array([0,0,1])

nd = np.array([1.2, 1.7, 1.6])
nd2 = nd**2

t1 = -ninit*np.cos(theta) + np.sqrt((1 -(ninit**2/nd2[2]**2)*np.sin(theta)**2)/(np.cos(phi)**2/nd2[0] + np.sin(phi)/nd2[1]**2))
t2 = -ninit*np.cos(theta) - np.sqrt((1 -(ninit**2/nd2[2]**2)*np.sin(theta)**2)/(np.cos(phi)**2/nd2[0] + np.sin(phi)/nd2[1]**2))

n1 = np.linalg.norm(ninit*kinit+t1*sigma)
n2 = np.linalg.norm(ninit*kinit+t2*sigma)
print(n1, n2)

k1 = np.array([np.cos(phi)*ninit*np.sin(theta)/n1, np.sin(phi)*ninit*np.sin(theta)/n1, (ninit*np.cos(theta) + t1)/n1])
print(np.linalg.norm(k1), k1, kinit)
k2 = np.array([np.cos(phi)*ninit*np.sin(theta)/n2, np.sin(phi)*ninit*np.sin(theta)/n2, (ninit*np.cos(theta) + t2)/n2])
print(np.linalg.norm(k2), k2)

print(check(k1, n1, nd))
print(check(k2, n2, nd))

