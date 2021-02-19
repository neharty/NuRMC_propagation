import numpy as np
import matplotlib.pyplot as plt

def curve(x,y):
    return (x*x + y*y) - 1

def grad(x,y):
    return [2*x, 2*y]

def ngrad(x,y):
    h = 1e-5
    dx = (curve(x+h,y) - curve(x-h,y))/(2*h)
    dy = (curve(x,y+h) - curve(x,y-h))/(2*h)
    return [dx, dy]

p1 = (np.sqrt(2), np.sqrt(2))
p2 = (1, 0)
p3 = (0, 1)

print(grad(*p1), ngrad(*p1))
print(grad(*p2), ngrad(*p2))
print(grad(*p3), ngrad(*p3))
