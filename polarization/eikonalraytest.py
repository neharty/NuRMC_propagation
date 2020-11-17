import numpy as np
from scipy.integrate import solve_ivp, simps
import polarizationfns as pl
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize, curve_fit, root, root_scalar
import pandas as pd
from tabulate import tabulate
from scipy.constants import speed_of_light
import snell_fns as sf

rmax = 1000
z0 = -300
zm = -200
dr = 10
dz = 10

theta0 = np.pi/3

# s wave test
def sodes(t, y):
    # theta, z, eikonal
    return [-(1/sf.ns(y[1]))*sf.dnsdz(y[1]), 1/np.tan(y[0]), sf.ns(y[1])**2/(sf.ns(z0)*np.sin(np.sin(theta0)))]

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

sol = solve_ivp(sodes, [0, rmax], [theta0, z0, 0], method='LSODA', max_step=0.1)

ax1.plot(sol.t, sol.y[1])
ax2.plot(sol.t, 2*np.pi*sol.y[2])
plt.show()
