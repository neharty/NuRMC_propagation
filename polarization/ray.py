import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult, fsolve
from scipy.constants import speed_of_light
from multiprocessing import Process, Queue
from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.utilities import medium

class ray:

    def __init__(self, x0, y0, z0, xf, yf, zf, ntype, raytype, eps, label=None):
        self.label = label

        self.x0, self.y0, self.z0 = x0, y0, z0
        self.xf, self.yf, self.zf = xf, yf, zf
        
        if raytype == 1 or raytype == 2:
            self.raytype = raytype
        else:
            print(raytype)
            raise RuntimeError('Please enter a valid ray type (1 or 2)')
        
        self.ntype = ntype
        self.eps = eps

        self.ray_x = []
        self.ray_y = []
        self.ray_z = []
        self.travel_time = 0.
        self.ray = OptimizeResult(t=[], y=[])
        self.launch_vector = []
        self.receive_vector = []
        self.initial_Efield = []
        self.final_Efield = []

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
    
    # NOTE: all functions are written with repect to the parameter s = arclength
    def ode(self, s, u):
        r = u[:3]
        p = u[3:-1]
        rdot = p/np.linalg.norm(p)
        n = self.n(r, rdot)
        gradrn = self.gradr(r, rdot)
        return np.array([p[0]/n, p[1]/n, p[2]/n, gradrn[0], gradrn[1], gradrn[2], n])
    
    def gradrdot(self, r, rdot):
        h = 1e-5
        Dxn = (self.n(r, [rdot[0] + h, rdot[1], rdot[2]]) - self.n(r, [rdot[0] - h, rdot[1], rdot[2]]))/(2*h)
        Dyn = (self.n(r, [rdot[0], rdot[1] + h, rdot[2]]) - self.n(r, [rdot[0], rdot[1] - h, rdot[2]]))/(2*h)
        Dzn = (self.n(r, [rdot[0], rdot[1], rdot[2] + h]) - self.n(r, [rdot[0], rdot[1], rdot[2] - h]))/(2*h)
        return np.array([Dxn, Dyn, Dzn])


    def gradr(self, r, rdot):
        h = 1e-5
        Dxn = (self.n([r[0] + h, r[1], r[2]], rdot) - self.n([r[0] - h, r[1], r[2]], rdot))/(2*h)
        Dyn = (self.n([r[0], r[1] + h, r[2]], rdot) - self.n([r[0], r[1] - h, r[2]], rdot))/(2*h)
        Dzn = (self.n([r[0], r[1], r[2] + h], rdot) - self.n([r[0], r[1], r[2] - h], rdot))/(2*h)
        return np.array([Dxn, Dyn, Dzn])
    
    def adj(self, A):
        #formula for a 3x3 matrix
        return 0.5*np.eye(3)*(np.trace(A)**2 - np.trace(A@A)) - np.trace(A)*A + A@A
    
    def n(self, r, rdot):
        tmp = self.eps(r)
        if not isinstance(tmp, np.ndarray):
            return np.sqrt(self.eps(r))
        else:
            rdot = rdot/np.linalg.norm(rdot)
            A = rdot @ self.eps(r) @ rdot
            B = rdot @ (np.trace(self.adj(self.eps(r)))*np.eye(3) - self.adj(self.eps(r))) @ rdot
            C = np.linalg.det(self.eps(r))
            A, B, C = np.longdouble(A), np.longdouble(B), np.longdouble(C)
            discr = (B + 2*np.sqrt(A*C))*(B - 2*np.sqrt(A*C))
            ntmp = np.sqrt((B + np.sqrt(np.abs(discr)))/(2*A))
            if self.ntype == 1:
                return np.sqrt(C/A)/ntmp
            if self.ntype == 2:
                return ntmp

    def shoot_ray(self, sf, phi0, theta0): 
        idir = np.array([np.cos(phi0)*np.sin(theta0), np.sin(phi0)*np.cos(theta0), np.cos(theta0)])
        dx0, dy0, dz0 = self.n([self.x0, self.y0, self.z0], idir)*idir
        solver = 'DOP853'
        mstep = min(sf, np.sqrt((self.xf - self.x0)**2+(self.yf-self.y0)**2 + (self.zf-self.z0)**2))/10
        sol=solve_ivp(self.ode, [0, sf], [self.x0, self.y0, self.z0, dx0, dy0, dz0, 0], method=solver, events=self.hit_top, max_step=30) 
        if len(sol.t_events[0]) == 0:
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            sinit = sol.t_events[0][0]
            evnt = sol.y_events[0][0]
            sol2 = solve_ivp(self.ode, [sinit, sf], [evnt[0], evnt[1], 0, evnt[3], evnt[4], -evnt[5], evnt[6]], method=solver, max_step=30)
            tvals = np.hstack((sol.t[sol.t < sinit], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < sinit])], sol2.y))
            return OptimizeResult(t=tvals, y=yvals)

    def _rootfn(self, args):
        #function for rootfinder
        sol = self.shoot_ray(args[0], args[1], args[2])
        #return (sol.y[0, -1] - self.xf)**2 +  (sol.y[1, -1] - self.yf)**2 + (sol.y[2, -1] - self.zf)**2
        return [sol.y[0, -1] - self.xf, sol.y[1, -1] - self.yf, sol.y[2, -1] - self.zf]

    def _get_ray(self, sf, phi, theta):
        if self.ray.t == []:
            minsol = root(self._rootfn, [sf, phi, theta])# options={'col_deriv': 0, 'xtol': 1e-10, 'ftol': 1e-10, 'gtol': 0.0, 'maxiter': 0, 'eps': 0.0, 'factor': 100, 'diag': None})
            print(minsol.success, minsol.message)
            self.ray = self.shoot_ray(minsol.x[0], minsol.x[1], minsol.x[2])
            
            self.launch_vector = self.ray.y[3:6, 0] / np.linalg.norm(self.ray.y[3:6, 0])
            self.receive_vector = self.ray.y[3:6, -1] / np.linalg.norm(self.ray.y[3:6, -1])
            
            self.initial_Efield = self.adj(self.eps((self.x0, self.y0, self.z0)) - self.n((self.x0, self.y0, self.z0), self.launch_vector)**2) @ (self.n((self.x0, self.y0, self.z0), self.launch_vector) * self.launch_vector)
            self.initial_Efield = self.initial_Efield / np.linalg.norm(self.initial_Efield)
            
            self.final_Efield = self.adj(self.eps((self.xf, self.yf, self.zf)) - self.n((self.xf, self.yf, self.zf), self.receive_vector)**2) @ (self.n((self.xf, self.yf, self.zf), self.receive_vector) * self.receive_vector)
            self.final_Efield = self.final_Efield / np.linalg.norm(self.final_Efield)
            #NOTE: E-field calculations are incorrect, only works when matrix is nonsingular (see Chen)
            return self.ray
        else:
            return self.ray
    
    def get_ray(self):
        if self.ray.t == []:
            sg, phig, thetag = self.get_guess()
            self._get_ray(sg, phig, thetag)
            print(self.xf, self.yf, self.zf, self.ray.y[0, -1], self.ray.y[1,-1], self.ray.y[2,-1])
            return self.ray
        else:
            return self.ray

    def get_guess(self, q=None):
        r0 = np.array([self.x0, self.y0, self.z0])
        rf = np.array([self.xf, self.yf, self.zf])
        g = analyticraytracing.ray_tracing(r0, rf, medium.get_ice_model('ARAsim_southpole'), n_frequencies_integration = 1)
        g.find_solutions()
        sg = g.get_path_length(self.raytype-1)
        lv = g.get_launch_vector(self.raytype-1)
        lv = lv/np.linalg.norm(lv)
        phig = np.arctan((self.yf - self.y0)/(self.xf-self.x0))
        thetag = np.arccos(lv[2])
        return sg, phig, thetag
        
        

        

