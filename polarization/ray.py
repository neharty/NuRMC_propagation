import numpy as np
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
from scipy.constants import speed_of_light
import importlib
import sys
from multiprocessing import Process, Queue

class ray:

    def __init__(self, x0, y0, z0, xf, yf, zf, raytype, eps, label = None):
        if label != None:
            self.label = label
        
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.xf, self.yf, self.zf = xf, yf, zf
        
        if raytype == 1 or raytype == 2:
            self.raytype = raytype
        else:
            print(raytype)
            raise RuntimeError('Please enter a valid ray type (1 or 2)')
        
        self.eps = eps

        self.ray_x = []
        self.ray_y = []
        self.ray_z = []
        self.launch_theta = 0.
        self.receive_theta = 0.
        self.travel_time = 0.
        self.ray = OptimizeResult(t=[], y=[])

    def copy_ray(self, odesol):
        self.odesol = odesol
        if odesol != (None, None):
            self.launch_angle = odesol.y[0,0]
            self.receive_angle = odesol.y[0,-1]
            self.travel_time = 1e9*odesol.y[2,-1]/speed_of_light
            self.ray_r = np.array(odesol.t)
            self.ray_z = np.array(odesol.y[1,:])

    def get_ray_parallel(self, q, sguess, phiguess, thetaguess):
        q.put(self.get_ray(sguess, phiguess, thetaguess))

    def hit_top(self, s, u):
        return u[2]

    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800
    
    # NOTE:all functions are written with repect to the parameter s = arclength
    
    def ode(self, s, u):
        r = u[:3]
        rdot = u[3:-1]
        grad_n = self.grad(r, rdot)
        n = self.n(r,rdot)
        return np.array([u[3]/n, u[4]/n, u[5]/n, grad_n[0], grad_n[1], grad_n[2], n])

    def grad(self, r, rdot):
        h=1e-5
        Dxn = (self.n([r[0] + h, r[1], r[2]], rdot) - self.n([r[0] - h, r[1], r[2]], rdot))/(2*h)
        Dyn = (self.n([r[0], r[1] + h, r[2]], rdot) - self.n([r[0], r[1] - h, r[2]], rdot))/(2*h)
        Dzn = (self.n([r[0], r[1], r[2] + h], rdot) - self.n([r[0], r[1], r[2] - h], rdot))/(2*h)
        return np.array([Dxn, Dyn, Dzn])
    
    def adj(self, A):
        #formula for a 3x3 matrix
        return 0.5*np.eye(3)*(np.trace(A)**2 - np.trace(A@A)) - np.trace(A)*A + A@A
    
    def n(self, r, rdot):
        rdot = rdot/np.linalg.norm(rdot)
        A = rdot @ self.eps(r) @ rdot
        B = rdot @ (np.trace(self.eps(r))*np.eye(3) - self.adj(self.eps(r))) @ rdot
        C = np.linalg.det(self.eps(r))
        discr = (B + 2*np.sqrt(A*C))*(B - 2*np.sqrt(A*C))
        ntmp = np.sqrt((B + np.sqrt(np.abs(discr)))/(2*A))
        if self.raytype == 1:
            return np.sqrt(C/A)/ntmp
        if self.raytype == 2:
            return ntmp

    def shoot_ray(self, sf, phi0, theta0):
        idir = np.array([np.cos(phi0)*np.sin(theta0), np.sin(phi0)*np.cos(theta0), np.cos(theta0)])
        dx0, dy0, dz0 = self.n([self.x0, self.y0, self.z0], idir)*idir
        
        solver = 'RK45'
        sol=solve_ivp(self.ode, [0, sf], [self.x0, self.y0, self.z0, dx0, dy0, dz0, 0], method=solver, events=self.hit_top, max_step=sf/10)
        if len(sol.t_events[0]) == 0:
            soll = OptimizeResult(t=sol.t, y=sol.y)
            #return OptimizeResult(t=sol.t, y=sol.y)
        else:
            sinit = sol.t_events[0][0]
            evnt = sol.y_events[0][0]
            sol2 = solve_ivp(self.ode, [sinit, sf], [evnt[0], evnt[1], 0, evnt[3], evnt[4], -evnt[5], evnt[6]], method=solver, max_step=sf/10)
            tvals = np.hstack((sol.t[sol.t < sinit], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < sinit])], sol2.y))
            soll = OptimizeResult(t=tvals, y=yvals)
            #return OptimizeResult(t=tvals, y=yvals)
        return soll
        #self.odesol = soll

    def _rootfn(self, args):
        #function for rootfinder
        sol = self.shoot_ray(args[0], args[1], args[2])
        return np.abs(sol.y[0:3, -1] - [self.xf, self.yf, self.zf])
    
    def get_ray(self, sf, phi, theta):
        minsol = root(self._rootfn, x0=np.array([sf, phi, theta]))
        print(minsol.success, minsol.message)
        odesol = self.shoot_ray(minsol.x[0], minsol.x[1], minsol.x[2])
        self.odesol = odesol
        return odesol

