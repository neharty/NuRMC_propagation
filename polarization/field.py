import numpy as np

class vector:

    v = np.array([x1,x2,x3])
    mag = np.linalg.norm(v)

    def __init__(self, a, b, c):
        self.x1 = a
        self.x2 = b
        self.x3 = c

    def __init__(self, w):
        self.x1 = w[0]
        self.x2 = w[1]
        self.x3 = w[2]
    
    def
