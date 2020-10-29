import numpy as np
from sympy import Matrix, Trace, N

# testing numerical example from https://doi.org/10.1364/JOSAA.12.002048

def adj(A):
    return np.array(Matrix(A).adjugate(), dtype=np.float64)

def get_ns(k,eps):
    k = k/np.linalg.norm(k)
    A = k.T@eps@k
    B = k.T@(adj(eps) - np.trace(adj(eps))*np.eye(3))@k
    C = np.linalg.det(eps)
    n1, n2 = np.sqrt(np.roots([A, B, C]))
    return n1, n2


def booker_roots(k, eps):
    k = k/np.linalg.norm(k)
    A = eps[2,2]
    B = 2*k[0]*eps[0,2]
    C = k[0]**2*(eps[0,0] + eps[2,2]) + (eps[0,2]**2 + eps[1,2]**2 - eps[2,2]*(eps[0,0]+eps[1,1]))
    D = 2*k[0]*((eps[0,1]*eps[1,2] - eps[0,2]*eps[1,1])+k[0]**2*eps[0,2])
    E = k[0]**4*eps[0,0] + k[0]**2*(eps[0,2]**2 + eps[0,1]**2-eps[0,0]*(eps[1,1]+eps[2,2]))+(eps[0,0]*eps[1,1]*eps[2,2] + 2*eps[0,1]*eps[0,2]*eps[1,2] - eps[0,0]*eps[1,2]**2 - eps[1,1]*eps[0,2]**2-eps[2,2]*eps[0,1]**2)
    return np.roots([A,B,C,D,E])

eps1 = np.array([[4.44228, 0, 1.09274],
                 [0, 2.89, 0],
                 [1.09274, 0, 1.83772]])

eps2 = np.array([[2.59918, -0.83615, 0.22880],
                 [-0.83615, 2.30894, -1.02415],
                 [0.22880, -1.02415, 4.26188]])

nui = np.pi/6
khati = np.array([np.sin(nui), 0, np.cos(nui)])
A = khati.T@eps1@khati
B = khati.T@(adj(eps1) - np.trace(adj(eps1))*np.eye(3))@khati
C = np.linalg.det(eps1)
n1, n2 = np.sqrt(np.roots([A, B, C]))
print('n1 and n2:', n1, n2)
ki = n2*khati

#region 1
A = eps1[2,2]
B = 2*khati[0]*eps1[0,2]
C = khati[0]**2*(eps1[0,0] + eps1[2,2]) + (eps1[0,2]**2 + eps1[1,2]**2 - eps1[2,2]*(eps1[0,0]+eps1[1,1]))
D = 2*khati[0]*((eps1[0,1]*eps1[1,2] - eps1[0,2]*eps1[1,1])+khati[0]**2*eps1[0,2])
E = khati[0]**4*eps1[0,0] + khati[0]**2*(eps1[0,2]**2 + eps1[0,1]**2-eps1[0,0]*(eps1[1,1]+eps1[2,2]))+(eps1[0,0]*eps1[1,1]*eps1[2,2] + 2*eps1[0,1]*eps1[0,2]*eps1[1,2] - eps1[0,0]*eps1[1,2]**2 - eps1[1,1]*eps1[0,2]**2-eps1[2,2]*eps1[0,1]**2)
kzs1 = np.roots([A,B,C,D,E])
kzs1 = kzs1[kzs1<0]
kr1 = np.array([ki[0], 0, kzs1[0]])
kr2 = np.array([ki[0], 0, kzs1[1]])

#region 2
A = eps2[2,2]
B = 2*khati[0]*eps2[0,2]
C = khati[0]**2*(eps2[0,0] + eps2[2,2]) + (eps2[0,2]**2 + eps2[1,2]**2 - eps2[2,2]*(eps2[0,0]+eps2[1,1]))
D = 2*khati[0]*((eps2[0,1]*eps2[1,2] - eps2[0,2]*eps2[1,1])+khati[0]**2*eps2[0,2])
E = khati[0]**4*eps2[0,0] + khati[0]**2*(eps2[0,2]**2 + eps2[0,1]**2-eps2[0,0]*(eps2[1,1]+eps2[2,2]))+(eps2[0,0]*eps2[1,1]*eps2[2,2] + 2*eps2[0,1]*eps2[0,2]*eps2[1,2] - eps2[0,0]*eps2[1,2]**2 - eps2[1,1]*eps2[0,2]**2- eps2[2,2]*eps2[0,1]**2)
kzs2 = np.roots([A,B,C,D,E])
kzs2 = kzs2[kzs2>0]
kt1 = np.array([ki[0], 0, kzs2[0]])
kt2 = np.array([ki[0], 0, kzs2[1]])

# e-field directions 
ei = adj(eps1 - n2**2*np.eye(3))@khati
print(ei)
print(ei/np.linalg.norm(ei))
