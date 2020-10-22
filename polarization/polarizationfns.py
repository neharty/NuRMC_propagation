import numpy as np
import scipy.linalg as spl

def get_n(s,n):
    '''
        computes eigenspeeds of birefringent 
        material as roots of a polynomial, from 
        Fresnel's equation

        s: direction (unit vector)
        n: array of speeds, n_i = sqrt(eps_i)

        NOTE: equation is only valid for diagonalized epsilon matrix
    '''

    sx, sy, sz = s[0], s[1], s[2]
    vx, vy, vz = 1/n[0], 1/n[1], 1/n[2]
    v1, v2 = np.sqrt(np.roots([sx**2+sy**2+sz**2, -sx**2*(vy**2+vz**2)-sy**2*(vx**2+vz**2)-sz**2*(vx**2+vy**2), (sx*vy*vz)**2+(sy*vx*vz)**2+(sz*vx*vy)**2]))
    return 1/v1, 1/v2

def get_null(s, eps, rcond=1e-8):
    '''
        computes null space of L(n,s) 
        matrix, which gives E polarizations
        
        s: direction (unit vector)
        eps: permittivity matrix
        rcond: tolerance for null space solver (see numpy doc)
    '''
    
    n1, n2 = get_n(s,np.sqrt(np.diag(eps)))
    s1, s2, s3 = s

    L1 = np.array([[eps[0,0] - n1**2*(s2**2+s3**2), eps[0,1] + n1**2*s1*s2, eps[0,2] + n1**2*s1*s3],
        [eps[1,0] + n1**2*s2*s1, eps[1,1] - n1**2*(s1**2+s3**2), eps[1,2] + n1**2*s2*s3], 
        [eps[2,0] + n1**2*s3*s1, eps[2,1] + n1**2*s3*s2, eps[2,2] - n1**2*(s1**2+s2**2)]])
    L2 = np.array([[eps[0,0] - n2**2*(s2**2+s3**2), eps[0,1] + n2**2*s1*s2, eps[0,2] + n2**2*s1*s3],
        [eps[1,0] + n2**2*s2*s1, eps[1,1] - n2**2*(s1**2+s3**2), eps[1,2] + n2**2*s2*s3],
        [eps[2,0] + n2**2*s3*s1, eps[2,1] + n2**2*s3*s2, eps[2,2] - n2**2*(s1**2+s2**2)]])
    return spl.null_space(L1, rcond=rcond), spl.null_space(L2, rcond=rcond)

def get_E(s, n, eps, rcond=1e-8):
    s1, s2, s3 = s
    
    n1 = n
    L1 = np.array([[eps[0,0] - n1**2*(s2**2+s3**2), eps[0,1] + n1**2*s1*s2, eps[0,2] + n1**2*s1*s3],
        [eps[1,0] + n1**2*s2*s1, eps[1,1] - n1**2*(s1**2+s3**2), eps[1,2] + n1**2*s2*s3],
        [eps[2,0] + n1**2*s3*s1, eps[2,1] + n1**2*s3*s2, eps[2,2] - n1**2*(s1**2+s2**2)]])
    
    return spl.null_space(L1, rcond=rcond)

def get_basis(eps):
    '''
        computes rotation matrix as 
        eigenvectors of polarization matrix

        eps: permittivity matrix
    '''

    ofd = eps[1,0]

    v1 = np.array([2*ofd, eps[1,1] - eps[0,0] - np.sqrt((eps[0,0]-eps[1,1])**2 + 4*ofd**2), 0])
    v1 = v1/spl.norm(v1)

    v2 = np.array([2*ofd, eps[1,1] - eps[0,0] + np.sqrt((eps[0,0]-eps[1,1])**2 + 4*ofd**2), 0])
    v2 = v2/spl.norm(v2)

    v3 = np.array([0.,0.,1.])
    
    R = np.zeros((3,3))
    R[:,0] = v1[:]
    R[:,1] = v2[:]
    R[:,2] = v3[:]

    #vals, vects = spl.eig(eps)
    #print(vects)

    return R

def rotate_basis(v, eps):
    '''
        
    '''

