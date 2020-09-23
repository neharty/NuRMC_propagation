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

print(p1)
print(p2)

dl = 0.1
c1 = n1*np.sin(theta0)
c2 = n2*np.sin(theta0)

ent = int(1e5)

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

for i in range(1, ent-1):
    n1s = np.sqrt(np.diag(eps(z1[i])))
    n2s = np.sqrt(np.diag(eps(z2[i])))
    n1 = n1s[1]
    n2 = n2s[0]*n2s[2]/np.sqrt(n2s[0]**2*np.sin(theta2)**2 +n2s[2]**2*np.cos(theta2)**2)

    theta1 = np.arcsin(c1/n1)
    theta2 = np.arcsin(c2/n2)
    print(theta2)
    if np.isnan(c2/n2):
        input()

    z1[i+1] = z1[i] + dl*np.cos(theta1)
    z2[i+1] = z2[i] + dl*np.cos(theta2)
    
    r1[i+1] = r1[i] + dl*np.sin(theta1)
    r2[i+1] = r2[i] + dl*np.sin(theta2)

plt.plot(r2, z2)
plt.plot(r1, z2)
plt.show()

