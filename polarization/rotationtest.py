import numpy as np
import matplotlib.pyplot as plt

eps = np.diag([3.1, 3.5, 3.8])
ofd = 0.5
eps[0, 1] = ofd
eps[1, 0] = ofd

vals, vects = np.linalg.eig(eps)
print(vals)
print((eps[0,0] +  eps[1,1] + np.sqrt((eps[0,0]-eps[1,1])**2 + 4*ofd**2))/2)
print((eps[0,0] +  eps[1,1] - np.sqrt((eps[0,0]-eps[1,1])**2 + 4*ofd**2))/2)
print(vects)
print(vects.T@eps@vects)
