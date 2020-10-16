import numpy as np
import matplotlib.pyplot as plt
import polarizationfns as pf

eps = np.diag([3.1, 3.3, 3.5])
ofd = 1e-2
eps[0, 1] = ofd
eps[1, 0] = ofd

R = pf.get_basis(eps)
print(R)
print(R.T@eps@R)

#initial direction
kinit = np.array([np.sin(np.pi/4), 0, np.cos(np.pi/4)])

#intial eps array (already diagonal)
epsinit = eps = np.diag([3.12, 3.3, 3.5])

#eigenspeeds (for s, p respectively)
n_s, n_p = pf.get_n(kinit, np.sqrt(np.diag(epsinit)))

#polarization vectors
p, s = pf.get_null(kinit, epsinit)

#p in plane, s out of plane
p = p.T[0]
s = s.T[0]
print(p)
print(s)

# plane of incidence is the x-z plane, plane normal is
pn = np.array([0,1,0])

#rotated k
kir = R@kinit

#rotated plane 
pnr = R@pn

#rotated s, p vectors
sr = R@s
pr = R@p

print(pr)
print(sr)
