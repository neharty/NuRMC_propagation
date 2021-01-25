import numpy as np

class vector:
    

    def __init__(self, w):
        self.x1 = w[0]
        self.x2 = w[1]
        self.x3 = w[2]
    
    def getVect(self):
        return np.array([self.x1, self.x2, self.x3])
    
    def getDir(self):
        return self.getVect()/self.getMag()

    def getMag(self):
        return np.linalg.norm(self.getVect())
