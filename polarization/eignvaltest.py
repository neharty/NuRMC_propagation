import numpy as np
import numpy.linalg as la
import polarizationfns as p

theta = np.random.random()*np.pi/2
phi = np.random.random()*2*np.pi

#theta = np.pi/4
#phi = 0

s = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

print('s:',s)

n2 = np.array([1.5, 1.7, 1.5])**2

eps = np.diag(n2)
epsinv = np.diag(1/n2)

'''
def get_ns(s, eps):
    s = s/la.norm(s)
    
    sm = np.array([[-(s[1]**2 + s[2]**2), s[0]*s[1], s[0]*s[2]],
    [s[1]*s[0], -(s[0]**2 + s[2]**2), s[1]*s[2]],
    [s[2] *s[0], s[2]*s[1], -(s[0]**2 + s[1]**2)]])
    
    inv = la.inv(eps)

    vals, vects = la.eig(inv.dot(sm))
    vals = vals[np.abs(vals) > 1e-16]
'''

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

vals, vects = get_eigns(s,eps)
print('eigenvals:',vals)
print('eigenvects:\n',vects)


plane_norm = np.cross(np.array([0,0,1]), s)
print(np.dot((eps@vects[:,0]).T, plane_norm))
print(np.dot((eps@vects[:,1]).T, plane_norm))
