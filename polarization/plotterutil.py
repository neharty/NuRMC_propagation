import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def nice8(z):
    a = 1.78
    b = -0.43
    c = 0.0132
    return a + b*np.exp(c*z)

fl = pd.read_csv('epsdata/evals_vs_depth.csv')
depth = -np.array(fl['Nominal Depth'])

eperp = 3.157
deltae = 0.034

#puts eigenvals in same coords as jordan et al.
n3, n2, n1 = np.sqrt(eperp + np.array(fl['E1'])*deltae), np.sqrt(eperp + np.array(fl['E2'])*deltae), np.sqrt(eperp + np.array(fl['E3'])*deltae)

n2n3avg = (n2+n3)/2
z0 = depth[0]

def test_func(z, a,b,c):
    return a + b/(1+np.exp(c*(z-z0)))

p0 = [min(n2n3avg-n1), max(n2n3avg-n1), 1]

params1, p = optimize.curve_fit(test_func, depth, n2n3avg - n1, p0, method='dogbox')

plt.semilogy(depth, n2n3avg - n1, '.', depth, test_func(depth, *params1), '-')
plt.show()
depth = np.linspace(0, -2800, num=280)
plt.plot(depth, nice8(depth), '-', depth, (1-test_func(depth, *params1))*nice8(depth), '--')
plt.show()

