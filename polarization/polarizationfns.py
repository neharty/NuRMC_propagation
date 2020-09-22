import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

n = np.array([1.5, 1.5, 1.7])

eps = np.diag(n**2)

s = np.array([1,0,0])
s=s/np.linalg.norm(s)

def get_n(s,n):
    sx, sy, sz = s[0], s[1], s[2]
    vx, vy, vz = 1/n[0], 1/n[1], 1/n[2]
    v1, v2 = np.sqrt(np.roots([sx**2+sy**2+sz**2, -sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2), (sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2]))
    return 1/v1, 1/v2

def get_p(s,n):
    n1, n2 = get_n(s,n)
    nx, ny, nz = n[0], n[1], n[2]
    e = 1e-12
    for ns in n:
        if np.abs(n1-ns) < e or np.abs(n2- ns) < e:
            sx, sy, sz = [1, 1, 1]
        else:
            sx, sy, sz = s[0], s[1], s[2]
    
    p1 = np.array([(n1**2-ny**2)*(n1**2-nz**2)*sx, (n1**2-nx**2)*(n1**2-nz**2)*sy, (n1**2-nx**2)*(n1**2-ny**2)*sz])
    p2 = np.array([(n2**2-ny**2)*(n2**2-nz**2)*sx, (n2**2-nx**2)*(n2**2-nz**2)*sy, (n2**2-nx**2)*(n2**2-ny**2)*sz])

    return p1/np.linalg.norm(p1), p2/np.linalg.norm(p2)

def get_null(s, eps):
    #n1, n2 = get_n(s,np.sqrt(np.diag(eps)))
    s1, s2, s3 = s

    n1 = np.sqrt(eps[0,0])
    n2 = 1/np.sqrt((s3**2/eps[0,0])+(s1**2/eps[2,2])) 

    L1 = np.array([[eps[0,0] - n1**2*(s2**2+s3**2), eps[0,1] + n1**2*s1*s2, eps[0,2] + n1**2*s1*s3],
        [eps[1,0] + n1**2*s2*s1, eps[1,1] - n1**2*(s1**2+s3**2), eps[1,2] + n1**2*s2*s3], 
        [eps[2,0] + n1**2*s3*s1, eps[2,1] + n1**2*s3*s2, eps[2,2] - n1**2*(s1**2+s2**2)]])
    L2 = np.array([[eps[0,0] - n2**2*(s2**2+s3**2), eps[0,1] + n2**2*s1*s2, eps[0,2] + n2**2*s1*s3],
        [eps[1,0] + n2**2*s2*s1, eps[1,1] - n2**2*(s1**2+s3**2), eps[1,2] + n2**2*s2*s3],
        [eps[2,0] + n2**2*s3*s1, eps[2,1] + n2**2*s3*s2, eps[2,2] - n2**2*(s1**2+s2**2)]])
    return spl.null_space(L1, rcond=1e-8), spl.null_space(L2, rcond=1e-8)

theta = np.linspace(0, np.pi/2, num=101)

x1 = np.zeros(len(theta))
y1 = np.zeros(len(theta))
z1 = np.zeros(len(theta))

x2 = np.zeros(len(theta))
y2 = np.zeros(len(theta))
z2 = np.zeros(len(theta))

for i in range(len(theta)):
    s = np.array([np.sin(theta[i]), 0, np.cos(theta[i])])
    p1, p2 = get_null(s, eps)
    p1, p2 = p1.T[0], p2.T[0]
    x1[i], y1[i], z1[i] = p1
    x2[i], y2[i], z2[i] = p2

fig, ax = plt.subplots(1,2)
ax[0].plot(theta, np.array([x1, y1, z1]).T)
ax[0].legend(('x1', 'y1', 'z1'))
ax[1].plot(theta, np.array([x2, y2, z2]).T)
ax[1].legend(('x2', 'y2', 'z2'))
plt.show()
