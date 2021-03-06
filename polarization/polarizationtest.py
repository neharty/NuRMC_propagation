import numpy as np
import scipy.linalg as spl
import polarizationfns as pl
import matplotlib.pyplot as plt

theta0 = np.pi/4
z0 = 0

s = np.array([np.sin(theta0), 0, np.cos(theta0)])
s = s/spl.norm(s)

L = 100

def n(z):
    return 2-z/L

def eps(z):
    return np.diag([1.5**2, 1.6**2, (n(z))**2])

n1, n2 = pl.get_n(s, np.diag(np.sqrt(eps(z0))))
p1, p2 = pl.get_null(s,eps(z0))

p1 = p1.T[0]
p2 = p2.T[0]

print(p1, np.linalg.norm(p1))
print(p2, np.linalg.norm(p2))

dl = 0.1
c1 = n1*np.sin(theta0)
c2 = n2*np.sin(theta0)

ent = int(2e3)

z1 = np.zeros(ent)
r1 = np.zeros(ent)

z2 = np.zeros(ent)
r2 = np.zeros(ent)

z1[1] = z1[0] + dl*np.cos(theta0)
z2[1] = z2[0] + dl*np.cos(theta0)

r1[1] = r1[0] + dl*np.sin(theta0)
r2[1] = r2[0] + dl*np.sin(theta0)

theta1 = theta0
theta2 = theta0

up1 = True
up2 = True

#compute wave 1
for i in range(1, ent-1):
    n1s = np.sqrt(np.diag(eps(z1[i])))
    n1 = n1s[1]
    
    if c1/n1>1 and up1:
        theta1 = 2*np.pi-theta2
        up1 = False
    elif not up1:
        theta1 = 2*np.pi-np.arcsin(c1/n1)
    else:
        theta1 = np.arcsin(c1/n1)

    z1[i+1] = z1[i] + dl*np.cos(theta1)
    r1[i+1] = r1[i] + dl*np.sin(theta1)

#compute wave 2
for i in range(1, ent-1):
    n2s = np.sqrt(np.diag(eps(z2[i])))
    n2tmp = n2
    n2 = n2s[0]*n2s[2]/np.sqrt(n2s[0]**2*np.sin(theta2)**2 +n2s[2]**2*np.cos(theta2)**2)

    if c2/n2>1 and up2:
        theta2 = 2*np.pi-theta2
        up2 = False
        z2[i+1] = z2[i] - dl*np.cos(theta2)
        r2[i+1] = r2[i] - dl*np.sin(theta2)
        p2 = -np.array([p2[0], p2[1], eps(z2[i])[2,2]*p2[2]/eps(z2[i-1])[2,2]])
    elif not up2:
        theta2 = 2*np.pi-np.arcsin(c2/n2)
        z2[i+1] = z2[i] - dl*np.cos(theta2)
        r2[i+1] = r2[i] - dl*np.sin(theta2)
        p2 = np.array([p2[0], p2[1], eps(z2[i])[2,2]*p2[2]/eps(z2[i-1])[2,2]])
    else:
        theta2 = np.arcsin(c2/n2)
        z2[i+1] = z2[i] + dl*np.cos(theta2)
        r2[i+1] = r2[i] + dl*np.sin(theta2)
        p2 = np.array([p2[0], p2[1], eps(z2[i])[2,2]*p2[2]/eps(z2[i-1])[2,2]])
    
    s = np.array([np.sin(theta2), 0, np.cos(theta2)])
    s = s/spl.norm(s)

print(p2, np.linalg.norm(p2))
print(z2[-1])
p1, p2 = pl.get_null(s,eps(z2[-1]))
print(p1.T[0],p2.T[0])

#plt.plot(r1, z1)
#plt.plot(r2, z2)
plt.show()

