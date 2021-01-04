import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def nice8(z):
    a = 1.78
    b = -0.43
    c = 0.0132
    return a + b*np.exp(-c*z)

fl = pd.read_csv('epsdata/evals_vs_depth.csv')
depth = np.array(fl['Nominal Depth'])

eperp = 3.157
deltae = 0.034

#puts eigenvals in same coords as jordan et al.
n3, n2, n1 = np.sqrt(eperp + np.array(fl['E1'])*deltae), np.sqrt(eperp + np.array(fl['E2'])*deltae), np.sqrt(eperp + np.array(fl['E3'])*deltae)

def test_func(z, a, b, c, d):
    return c + a*(z-b)**d

'''
def test_func(z, z0, a,b,c):
    return a + b*np.exp(-(z-z0)/c)
'''
params1, p = optimize.curve_fit(test_func, depth, n1 - nice8(depth))

params2, p = optimize.curve_fit(test_func, depth, n2 - nice8(depth))

params3, p = optimize.curve_fit(test_func, depth, n3 - nice8(depth))

plt.loglog(depth, n1 - nice8(depth), '.', depth, n2 - nice8(depth), '.', depth, n3 - nice8(depth), '.')
plt.show()
plt.plot(depth, n1 - nice8(depth), '.', depth, test_func(depth, *params1), '--')
plt.show()
plt.plot(depth, n2 - nice8(depth), '.', depth, test_func(depth, *params2), '--')
plt.show()
plt.plot(depth, n3 - nice8(depth), '.', depth, test_func(depth, *params3), '--')
plt.show()

