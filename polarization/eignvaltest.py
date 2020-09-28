import numpy as np
import numpy.linalg as la

theta = np.pi/4
phi = 0

s = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

print(s)

sm = np.array([[-(s[1]**2 + s[2]**2), s[0]*s[1], s[0]*s[2]],
    [s[1]*s[0], -(s[0]**2 + s[2]**2), s[1]*s[2]],
    [s[2] *s[0], s[2]*s[1], -(s[0]**2 + s[1]**2)]])

eps = np.diag(np.array([1.78, 1.6, 1.5])**2)

def get_ns(s, eps):
    s = s/la.norm(s)
    
    sm = np.array([[-(s[1]**2 + s[2]**2), s[0]*s[1], s[0]*s[2]],
    [s[1]*s[0], -(s[0]**2 + s[2]**2), s[1]*s[2]],
    [s[2] *s[0], s[2]*s[1], -(s[0]**2 + s[1]**2)]])
    
    inv = la.inv(eps)

    vals, vects = la.eig(inv.dot(sm))
    vals = vals[np.abs(vals) > 1e-16]
    print(np.sqrt(1/-vals))


get_ns(s,eps)

