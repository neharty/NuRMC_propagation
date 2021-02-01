import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult

d = 1e-16
n1 = 1.3
n2 = 0

'''
    def n(z):
        #return 2-0.5*np.exp(z/100)
        if z < 100 - d:
            return 1
        if z >= 100 - d and z <= 100+d:
            return (n2-n1)*z/(2*d) + n2 - (100+d)*(n2-n1)/(2*d)
        if z > 100 + d:
            return n2

    def dndz(z):
        #return -0.5/100 * np.exp(z/100)
        if np.abs(100-z) <= d:
            return (n2-n1)/(2*d)
        else:
            return 0
'''

def n(z):
    if z>100:
        print(z)
    return 1

def dndz(z):
    return 0

def odes(t, y):
    return [-dndz(y[1])/n(y[1]), 1/np.tan(y[0])]

def hit_top(t, y):
    return y[1]-100

hit_top.terminal=True

def get_ray(odefn, event):
    sol=solve_ivp(odes, [0, 200], [np.pi/4, 0], method='LSODA', events=event)
    if sol.t_events == None:
        return sol
    else:
        tinit = sol.t_events[0][0]
        #print(tinit)
        thetainit = sol.y_events[0][0][0]
        #print(thetainit)
        zinit = sol.y_events[0][0][1]
        #print(np.pi-thetainit, zinit)
        sol2 = solve_ivp(odes, [tinit, 200], [np.pi-thetainit, zinit], method='LSODA', events=event)
        tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
        #print(tvals)
        #print(sol2.y)
        yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
        #print(yvals)
        return OptimizeResult(t=tvals, y=yvals)

sol = get_ray(odes, hit_top)

plt.plot(sol.t, sol.y[1])
plt.xlabel('r')
plt.ylabel('z')
plt.ylim([0,200])
plt.show()
#iplt.clf()
#plt.plot(sol.y[0], sol.y[1])
#plt.show()
