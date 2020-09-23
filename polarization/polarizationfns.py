import numpy as np
import scipy.linalg as spl

def get_n(s,n):
    sx, sy, sz = s[0], s[1], s[2]
    vx, vy, vz = 1/n[0], 1/n[1], 1/n[2]
    v1, v2 = np.sqrt(np.roots([sx**2+sy**2+sz**2, -sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2), (sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2]))
    return 1/v1, 1/v2

def get_null(s, eps, rcond=1e-8):
    n1, n2 = get_n(s,np.sqrt(np.diag(eps)))
    s1, s2, s3 = s

    L1 = np.array([[eps[0,0] - n1**2*(s2**2+s3**2), eps[0,1] + n1**2*s1*s2, eps[0,2] + n1**2*s1*s3],
        [eps[1,0] + n1**2*s2*s1, eps[1,1] - n1**2*(s1**2+s3**2), eps[1,2] + n1**2*s2*s3], 
        [eps[2,0] + n1**2*s3*s1, eps[2,1] + n1**2*s3*s2, eps[2,2] - n1**2*(s1**2+s2**2)]])
    L2 = np.array([[eps[0,0] - n2**2*(s2**2+s3**2), eps[0,1] + n2**2*s1*s2, eps[0,2] + n2**2*s1*s3],
        [eps[1,0] + n2**2*s2*s1, eps[1,1] - n2**2*(s1**2+s3**2), eps[1,2] + n2**2*s2*s3],
        [eps[2,0] + n2**2*s3*s1, eps[2,1] + n2**2*s3*s2, eps[2,2] - n2**2*(s1**2+s2**2)]])
    return spl.null_space(L1, rcond=rcond), spl.null_space(L2, rcond=rcond)

