import numpy as np
cimport numpy
from scipy.constants import speed_of_light

# ice parameters
cdef double nss = 1.35
cdef double nd = 1.78
cdef double c = 0.0132

cdef double phi = np.pi*2*np.random.random()

def odefns(double t, object y, int raytype, param='r'):
    # odes 
    if raytype ==1:
        ntype = npp
    elif raytype ==2:
        ntype = ns
    else:
        raise RuntimeError('Please enter a valid ray type (1 or 2)')

    if param == 'r':
        # form is [d(theta)/dr, dzdr, dtdr], r = radial distance
        return [-np.cos(y[0])*zderiv(y[1], phi, y[0], ntype)/(ntype(y[1], phi, y[0])*np.cos(y[0])+thetaderiv(y[1],phi, y[0], ntype)*np.sin(y[0])), 1/np.tan(y[0]), ntype(y[1], phi, y[0])/np.abs(np.sin(y[0]))]
    if param == 'l':
        # form is [d(theta)/ds, dzds, dtds, drds]
        return [-np.sin(y[0])*np.cos(y[0])*zderiv(y[1], phi, y[0], ntype)/(ntype(y[1], phi, y[0])*np.cos(y[0])+thetaderiv(y[1],phi, y[0], ntype)*np.sin(y[0])), np.cos(y[0]), ntype(y[1], phi, y[0]), np.sin(y[0])]

#sim model parameters
cdef double cont = 0.9

cdef double n1(double z):
    # x index of refraction function
    # extraordinary index of refraction function
    # from nice8 ARA model

    cdef double a = nd
    cdef double b = nss - nd
    return a + b*np.exp(z*c)

cdef double n2(double z):
    # y index of refraction function
    #same as n1 for testing
    return n1(z)

cdef double n3(double z):
    # z index of refraction fn
    return cont*n1(z)

cpdef object eps(double z):
    # epsilon is diagonal
    return np.diag([(n1(z))**2, (n2(z))**2, (n3(z))**2])

cdef double A(double z, double phi, double theta):
    return np.sin(theta)**2*(n1(z)**2*np.cos(phi)**2 + n2(z)**2*np.sin(phi)**2) + n3(z)**2*np.cos(theta)**2
    
cdef double B(double z, double phi, double theta):
    return ((n1(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.cos(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n2(z)**2)*np.sin(phi)**2*np.sin(theta)**2 + (n2(z)**2*n3(z)**2 + n1(z)**2*n3(z)**2)*np.cos(theta)**2)

cdef double C(double z, double phi, double theta):
    return (n1(z)*n2(z)*n3(z))**2

def ns(double z, double phi, double theta):
    #s-polarization index of refraction
    #if z >=0:
    #    print(z)
    #a = np.longdouble(A(z, phi, theta))
    #b = np.longdouble(B(z, phi, theta))
    #c = np.longdouble(C(z, phi, theta))
    # from https://doi.org/10.1137/1.9780898718027
    # redone in 10.1090/S0025-5718-2013-02679-8
    #w = 4*a*c
    #e = fma(-c, 4*a, w)
    #f = fma(b, b, -w)
    #discr = f + e
    cdef double a = np.longdouble(A(z, phi, theta))
    cdef double b = np.longdouble(B(z, phi, theta))
    cdef double c = np.longdouble(C(z, phi, theta))
    cdef double discr = (b + 2*np.sqrt(a*c))*(b - 2*np.sqrt(a*c))
    return np.sqrt((b + np.sqrt(np.abs(discr)))/(2*a))

def npp(double z, double phi, double theta):
    #p-polarization index of refraction
    return np.sqrt(C(z, phi, theta)/(A(z, phi, theta)))/ns(z, phi, theta)

def zderiv(double z, double phi, double theta, n):
    cdef double h=1e-5
    return (n(z + h, phi, theta) - n(z-h, phi, theta))/(2*h)

def thetaderiv(double z, double phi, double theta, n):
    cdef double h=1e-5
    return (n(z, phi, theta + h) - n(z, phi, theta - h))/(2*h)

#for testing against actual values
cdef double ns_a(double z, double phi, double theta):
    return n1(z)

cdef double npp_a(double z, double phi, double theta):
    return n1(z)*n3(z)/np.sqrt(n3(z)**2*np.cos(theta)**2+n1(z)**2*np.sin(theta)**2)



