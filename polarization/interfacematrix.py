import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def get_eigns(s, eps):
    s = s/la.norm(s)

    sm = crossm(s)

    vals, vects = la.eig(epsinv.dot(sm@sm))
    mask = np.abs(vals)>1e-12
    vals = 1/np.sqrt(-vals[mask])
    vects = vects[:, mask]
    return vals, vects

def crossm(a):
    #cross product matrix from a vector
    a1, a2, a3 = a
    A = np.zeros((3,3))
    A[0,1] = -a3
    A[0,2] = a2
    A[1,2] = -a1

    A = A - A.T
    return A


s = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

print('s:',s1)

n2 = np.array([1.5, 1.7, 1.5])**2

eps = np.diag(n2)
epsinv = np.diag(1/n2)

