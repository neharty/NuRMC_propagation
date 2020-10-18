import numpy as np
import matplotlib.pyplot as plt
import polarizationfns as pf

def normalize(x):
    return x/np.linalg.norm(x)

eps = np.diag([3.1, 3.3, 3.5])
ofd = 1e-1
eps[0, 1] = ofd
eps[1, 0] = ofd

R = pf.get_basis(eps)

#initial direction
theta = np.random.random()*2*np.pi
phi = np.random.random()*np.pi/2

kinit = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
ptest = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
stest = np.array([np.sin(phi)*(np.cos(theta)**2 - np.sin(theta)**2), np.cos(phi), 0])

#intial eps array (already diagonal)
epsinit = np.diag([3.12, 3.3, 3.5])

#eigenspeeds (for v1, v2 respectively)
n1, n2 = pf.get_n(kinit, np.sqrt(np.diag(epsinit)))

#polarization vectors
v1, v2 = pf.get_null(kinit, epsinit)
v1 = normalize(v1.T[0])
v2 = normalize(v2.T[0])

D1 = normalize(epsinit@v1)
D2 = normalize(epsinit@v2)

ntmp1 = D1[0]**2/epsinit[0,0] + D1[1]**2/epsinit[1,1] + D1[2]**2/epsinit[2,2]
ntmp2 = D2[0]**2/epsinit[0,0] + D2[1]**2/epsinit[1,1] + D2[2]**2/epsinit[2,2]

# plane of incidence is the x-z plane, plane normal is
# sigma is interface normal
sigma = np.array([0,0,1])
pn = normalize(np.cross(kinit,sigma))

#calculate s, p components of D1, D2
s1 = pn.dot(D1)*pn
p1 = D1 - s1

s2 = pn.dot(v2)*pn
p2 = v2 - s2

#p in plane, s out of plane
print('p1 . pn = ', p1.dot(pn))
print('s1 . pn = ', normalize(s1).dot(pn))

print('p2 . pn = ', p2.dot(pn))
print('s2 . pn = ', normalize(s2).dot(pn))

# ROTATION TESTS
#rotated k
kir = R@kinit

#rotated plane 
pnr = R@pn

#rotated s, p vectors
sr1 = R@s1
pr1 = R@p1

sr2 = R@s2
pr2 = R@p2


