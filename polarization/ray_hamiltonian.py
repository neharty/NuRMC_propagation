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
        if odesol != (None, None):
            self.ray = odesol
            self.travel_time = 1e9*odesol.y[-1,-1]/speed_of_light

    def get_ray_parallel(self, q, sguess, phiguess, thetaguess):
        q.put(self.get_ray(sguess, phiguess, thetaguess)) 

    def hit_top(self, s, u):
        return u[2]
    
    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800
    
    # NOTE: all functions are written with repect to the parameter s = arclength
    def ode(self, s, u):
        q = u[:3]
        p = u[3:-1]
        rdot = p/np.linalg.norm(p, 2)
        n = self.n(q, rdot)
        qdot = self.DpH(q, p)
        qdn = np.linalg.norm(qdot, 2)
        qdot = qdot/qdn
        pdot = -self.DqH(q, p)/qdn
        return np.array([qdot[0], qdot[1], qdot[2], pdot[0], pdot[1], pdot[2], np.dot(p, qdot) if (np.arccos(np.dot(rdot, qdot)) < np.pi/2 and np.arccos(np.dot(rdot, qdot)) >= 0) else np.dot(-p, qdot)])

    def grad(self, r, rdot):
        h = 1e-5
        Dxn = (self.n([r[0] + h, r[1], r[2]], rdot) - self.n([r[0] - h, r[1], r[2]], rdot))/(2*h)
        Dyn = (self.n([r[0], r[1] + h, r[2]], rdot) - self.n([r[0], r[1] - h, r[2]], rdot))/(2*h)
        Dzn = (self.n([r[0], r[1], r[2] + h], rdot) - self.n([r[0], r[1], r[2] - h], rdot))/(2*h)
        return np.array([Dxn, Dyn, Dzn])

    def H(self, q, p):
        D = np.array([[0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]])
        return np.longdouble(np.linalg.det(D@D + self.eps(q)))

    def DqH(self, q, p):
        h = 1e-5
        DxH = (self.H([q[0] + h, q[1], q[2]], p) - self.H([q[0] - h, q[1], q[2]], p))/(2*h)
        DyH = (self.H([q[0], q[1] + h, q[2]], p) - self.H([q[0], q[1] - h, q[2]], p))/(2*h)
        DzH = (self.H([q[0], q[1], q[2] + h], p) - self.H([q[0], q[1], q[2] - h], p))/(2*h)
        return np.array([DxH, DyH, DzH])
    
    def DpH(self, q, p):
        h = 1e-5
        Dp1H = (self.H(q, [p[0] + h, p[1], p[2]]) - self.H(q, [p[0] - h, p[1], p[2]]))/(2*h)
        Dp2H = (self.H(q, [p[0], p[1] + h, p[2]]) - self.H(q, [p[0], p[1] - h, p[2]]))/(2*h)
        Dp3H = (self.H(q, [p[0], p[1], p[2] + h]) - self.H(q, [p[0], p[1], p[2] - h]))/(2*h)
        return np.array([Dp1H, Dp2H, Dp3H])

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
            #A, B, C = np.longdouble(A), np.longdouble(B), np.longdouble(C)
            discr = (B + 2*np.sqrt(A*C))*(B - 2*np.sqrt(A*C))
            ntmp = np.sqrt((B + np.sqrt(np.abs(discr)))/(2*A))
            
            no = np.sqrt(self.eps(r)[0,0])
            ne = np.sqrt(self.eps(r)[2,2])
            theta = np.arccos(rdot[2])

            if self.ntype == 1:
                #return no*ne/np.sqrt(ne**2*np.cos(theta)**2+no**2*np.sin(theta)**2)
                return np.sqrt(C/A)/ntmp
            if self.ntype == 2:
                #print(np.abs(ntmp - no))
                return ntmp
                #return no

    def shoot_ray(self, sf, phi0, theta0): 
        idir = np.array([np.cos(phi0)*np.sin(theta0), np.sin(phi0)*np.sin(theta0), np.cos(theta0)])
        dx0, dy0, dz0 = self.n([self.x0, self.y0, self.z0], idir)*idir
        solver = 'RK45'
        mstep = max(np.abs(sf), np.sqrt((self.xf - self.x0)**2+(self.yf-self.y0)**2 + (self.zf-self.z0)**2))/30
        sol=solve_ivp(self.ode, [0, sf], [self.x0, self.y0, self.z0, dx0, dy0, dz0, 0], method=solver, events=self.hit_top, max_step=mstep) 
        if len(sol.t_events[0]) == 0:
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            sinit = sol.t_events[0][0]
            evnt = sol.y_events[0][0]
            sol2 = solve_ivp(self.ode, [sinit, sf], [evnt[0], evnt[1], 0, evnt[3], evnt[4], -evnt[5], evnt[6]], method=solver, max_step=mstep)
            tvals = np.hstack((sol.t[sol.t < sinit], sol2.t))
            #yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < sinit])], sol2.y))
            yvals = np.hstack((sol.y, sol2.y))
            return OptimizeResult(t=tvals, y=yvals)

    def _rootfn(self, args):
        #function for rootfinder
        sol = self.shoot_ray(args[0], args[1], args[2])
        #return (sol.y[0, -1] - self.xf)**2 +  (sol.y[1, -1] - self.yf)**2 + (sol.y[2, -1] - self.zf)**2
        return [sol.y[0, -1] - self.xf, sol.y[1, -1] - self.yf, sol.y[2, -1] - self.zf]

    def _get_ray(self, sf, phi, theta):
        if self.ray.t == []:
            minsol = root(self._rootfn, [sf, phi, theta], options={'xtol': 1e-12, 'maxfev': 0, 'band': None, 'eps': None, 'factor': 100, 'diag': None})
            print(minsol.success, minsol.message)
            self.copy_ray(self.shoot_ray(minsol.x[0], minsol.x[1], minsol.x[2]))
            ''' 
            self.launch_vector = self.ray.y[3:6, 0] / np.linalg.norm(self.ray.y[3:6, 0])
            self.receive_vector = self.ray.y[3:6, -1] / np.linalg.norm(self.ray.y[3:6, -1])
            
            self.initial_Efield = self.adj(self.eps((self.x0, self.y0, self.z0)) - self.n((self.x0, self.y0, self.z0), self.launch_vector)**2) @ (self.n((self.x0, self.y0, self.z0), self.launch_vector) * self.launch_vector)
            self.initial_Efield = self.initial_Efield / np.linalg.norm(self.initial_Efield)
            
            self.final_Efield = self.adj(self.eps((self.xf, self.yf, self.zf)) - self.n((self.xf, self.yf, self.zf), self.receive_vector)**2) @ (self.n((self.xf, self.yf, self.zf), self.receive_vector) * self.receive_vector)
            self.final_Efield = self.final_Efield / np.linalg.norm(self.final_Efield)
            #NOTE: E-field calculations are incorrect, only works when matrix is nonsingular (see Chen)
            '''
            return self.ray
        else:
            return self.ray
    
    def get_ray(self, sg, phig, thetag):
        if self.ray.t == []:
            #sg, phig, thetag = self.get_guess()
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
        #sg = g.get_travel_time(self.raytype-1)
        sg = g.get_path_length(self.raytype-1)
        #print('\nguess length:', sg, 'guess time', g.get_travel_time(self.raytype-1))
        self.guess_l = sg
        self.guess_t = g.get_travel_time(self.raytype-1)
        lv = g.get_launch_vector(self.raytype-1)
        lv = lv/np.linalg.norm(lv)
        phig = np.arctan((self.yf - self.y0)/(self.xf-self.x0))
        thetag = np.arccos(lv[2])
        return sg, phig, thetag
        
class rays(ray):

    def __init__(self, x0, y0, z0, xf, yf, zf, eps, dr = None):
        #naming convention is r_ik, i = ntype, k = raytype
        
        self.r = np.array([[None, None], [None, None]])
        for i in [1,2]:
            for k in [1,2]:
                self.r[i-1, k-1] = ray(x0, y0, z0, xf, yf, zf, i, k, eps)
        
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.eps = eps
        self.get_rays()

    def set_guess(self):
        r0 = np.array([self.x0, self.y0, self.z0])
        rf = np.array([self.xf, self.yf, self.zf])
        g = analyticraytracing.ray_tracing(r0, rf, medium.get_ice_model('ARAsim_southpole'), n_frequencies_integration = 1)
        g.find_solutions()
        self.sg1, self.sg2 = g.get_path_length(0), g.get_path_length(1)
        self.guess_l1 = self.sg1, self.sg2
        self.guess_t1, self.guess_t2 = g.get_travel_time(0), g.get_travel_time(1)
        lv1, lv2 = g.get_launch_vector(0), g.get_launch_vector(1)
        lv1, lv2 = lv1/np.linalg.norm(lv1), lv2/np.linalg.norm(lv2)
        self.phig = np.arctan((self.yf - self.y0)/(self.xf-self.x0))
        self.thetag1, self.thetag2 = np.arccos(lv1[2]), np.arccos(lv2[2])
    
    def get_guess(self, raytype):
        if raytype == 1:
            return (self.sg1, self.phig, self.thetag1)
        if raytype == 2:
            return (self.sg2, self.phig, self.thetag2)

    def get_rays(self, par=True):
        if par == True:
            self.set_guess()
            q = np.array([[Queue() for k in [1,2]] for i in [1,2]])

            p = np.array([[Process(target=self.r[i-1,k-1].get_ray_parallel, args=(q[i-1, k-1], *self.get_guess(k))) for k in [1,2]] for i in [1,2]])

            [[p[i,k].start() for i in range(2)] for k in range(2)]
            [[p[i,k].join() for i in range(2)] for k in range(2)]

            for i in [0,1]:
                for k in [0,1]:
                    self.r[i,k].copy_ray(q[i,k].get())

    def get_ray(self, i, k):
        return self.r[i,k].ray

    def get_time(self, i, k):
        return self.r[i,k].travel_time
