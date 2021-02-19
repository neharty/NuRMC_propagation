import numpy as np
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
from scipy.constants import speed_of_light
import importlib
import sys
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Process, Queue
from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.utilities import medium

class ray:

    def get_ray_r(self):
        return self.ray_r
    
    def get_ray_z(self):
        return self.ray_z
    
    def get_launch_angle(self):
        return self.launch_angle
    
    def get_receive_angle(self):
        return self.receive_angle

    def get_travel_time(self):
        return self.travel_time

    def __init__(self, z0, zf, rf, phi, eps, ntype, raytype, label = None, dr = None):
        self.label = label
        self.rf = rf
        self.phi = phi
        self.z0 = z0
        self.zf = zf
        self.eps = eps

        if dr == None:
            self.dr = self.rf/20
        else:
            self.dr = min(dr, rf/20)
        
        self.raytype = raytype
        self.ntype = ntype
        self.ray_r = []
        self.ray_z = []
        self.launch_angle = 0.
        self.travel_time = 0.
        self.ray = OptimizeResult(t=[], y=[])
    
    def copy_ray(self, odesol):
        self.ray = ray
        if odesol != (None, None):
            self.launch_angle = odesol.y[0,0]
            self.receive_angle = odesol.y[0,-1]
            self.travel_time = 1e9*odesol.y[2,-1]/speed_of_light
            self.ray_r = np.array(odesol.t)
            self.ray_z = np.array(odesol.y[1,:])

    def get_ray_parallel(self, q, sg, thetag):
        q.put(self.get_ray(sg, thetag))

    def hit_top(self, t, y):
        return y[1]

    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800
 
    def ode(self, r, u):
        q = u[:3]
        p = u[3:-1]
        rdot = p/np.linalg.norm(p, 2)
        qdot = self.DpH(q, p)
        qdn = np.linalg.norm(qdot, 2)
        qdot = qdot/qdn
        pdot = -self.DqH(q, p)/qdn
        return np.array([qdot[0], qdot[1], qdot[2], pdot[0], pdot[1], pdot[2], np.dot(p, qdot) if (np.arccos(np.dot(rdot, qdot)) < np.pi/2 and np.arccos(np.dot(rdot, qdot)) >= 0) else np.dot(-p, qdot)])

    def H(self, q, p):
        D = np.array([[0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]])
        epsq = np.longdouble(self.eps(q))
        return np.linalg.det(D@D + self.eps(q))

    def DqH(self, q, p):
        h = 1e-5 
        DzH = (self.H([q[0], q[1], q[2] + h], p) - self.H([q[0], q[1], q[2] - h], p))/(2*h)
        return np.array([0, 0, DzH])
    
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
            if self.ntype == 1:
                return np.sqrt(C/A)/ntmp
            if self.ntype == 2:
                return ntmp

    def shoot_ray(self, s, theta0):
        solver = 'RK45'
        #dense_output = True
        idir = np.array([np.cos(self.phi)*np.sin(theta0), np.sin(self.phi)*np.sin(theta0), np.cos(theta0)])
        dx0, dy0, dz0 = self.n([0, 0, self.z0], idir)*idir
        #self.pconst = self.n([0, 0, self.z0], idir)*np.sin(theta0)
        #print(self.n([0, 0, self.z0], idir))
        sol=solve_ivp(self.ode, [0, s], [0, 0, self.z0, dx0, dy0, dz0, 0], method=solver, events=self.hit_top, max_step=50)
        if len(sol.t_events[0]) == 0:
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            tinit = sol.t_events[0][0]
            evnt = sol.y_events[0][0]
            travtime = sol.y_events[0][0][2]
            sol2 = solve_ivp(self.ode, [tinit, s], [evnt[0], evnt[1], evnt[2], evnt[3], evnt[4], -evnt[5], evnt[6]], method=solver, max_step=50)
            tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
            return OptimizeResult(t=tvals, y=yvals)

    def _rootfn(self, args):
        #function for rootfinder
        sol = self.shoot_ray(*args)
        return [np.sqrt((sol.y[0, -1] - self.rf*np.cos(self.phi))**2 + (sol.y[1, -1] - self.rf*np.sin(self.phi))**2), sol.y[2, -1] - self.zf]
    
    def get_ray_1guess(self, thetag):
        lb, rb = self.get_bounds(thetag)
        if(lb == None and rb == None):
            return None, None
        else:
            minsol = root_scalar(minfn, bracket=[lb,rb])

        print(minsol.converged, minsol.flag)
        odesol = self.shoot_ray(minsol.root)
        return odesol
    
    def get_ray(self, sg, thetag):
        if self.ray.t == []:
            #lb, rb = self._get_bounds(thetag)
            self._get_ray(sg, thetag)
            self.travel_time = 1e9*self.ray.y[-1,-1]/speed_of_light
            #print(self.xf, self.yf, self.zf, self.ray.y[0, -1], self.ray.y[1,-1], self.ray.y[2,-1])
            return self.ray
        else:
            return self.ray

    def _get_ray(self, sf, theta):
        if self.ray.t == []:
            #print(bounds)
            minsol = root(self._rootfn, [sf, theta])
            print(minsol.success, minsol.message)
            self.ray = self.shoot_ray(*minsol.x)
            '''
            self.initial_Efield = self.adj(self.eps((self.x0, self.y0, self.z0)) - self.n((self.x0, self.y0, self.z0), self.launch_vector)**2) @ (self.n((self.x0, self.y0, self.z0), self.launch_vector) * self.launch_vector)
            self.initial_Efield = self.initial_Efield / np.linalg.norm(self.initial_Efield)

            self.final_Efield = self.adj(self.eps((self.xf, self.yf, self.zf)) - self.n((self.xf, self.yf, self.zf), self.receive_vector)**2) @ (self.n((self.xf, self.yf, self.zf), self.receive_vector) * self.receive_vector)
            self.final_Efield = self.final_Efield / np.linalg.norm(self.final_Efield)
            #NOTE: E-field calculations are incorrect, only works when matrix is nonsingular (see Chen)
            '''
            return self.ray
        else:
            return self.ray


    def _get_bounds(self, initguess, xtol = None, maxiter=None):
        if xtol != None:
            xtol = xtol
        else:
            xtol=1e-4

        if maxiter != None:
            maxiter = maxiter
        else:
            maxiter=200

        dxi=1e-2
        dx = dxi
        zendintl = self._rootfn(initguess)
        dz = np.sign(zendintl)
        grad = np.sign(self._rootfn(initguess + dx) - zendintl)
        print(zendintl, dz, grad)
        zendnew = zendintl
        inum = 0
        lastguess = initguess
        newguess = lastguess
        while dz == np.sign(zendnew) and newguess < np.pi/2 and newguess > 0:
            lastguess = newguess
            if (dz > 0 and grad > 0) or (dz < 0 and grad < 0):
                newguess -= dx
            if (dz > 0 and grad < 0) or (dz < 0 and grad > 0):
                newguess += dx
            zendnew = self._rootfn(newguess)

        if newguess >= np.pi/2 or newguess <= 0:
            print('ERROR: no interval found')
            print(np.sign(zendintl), newguess)
            return None, None
        else:
            lb, rb = lastguess, newguess

        print('initial guess:', initguess, 'returned bounds:', lb, rb)
        if lb < rb:
            return lb, rb
        else:
            return rb, lb

class rays2D(ray):
    
    def __init__(self, x0, y0, z0, xf, yf, zf, eps, dr = None):
        #naming convention is r_ik, i = ntype, k = raytype
        
        self.r = np.array([[None, None], [None, None]])
        for i in [1,2]:
            for k in [1,2]:
                self.r[i-1, k-1] = ray(z0, zf, np.sqrt((xf- x0)**2 + (yf-y0)**2), np.arctan((yf-y0)/(xf-x0)), eps, i, k)
        
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
        self.thetag1, self.thetag2 = np.arccos(lv1[2]), np.arccos(lv2[2])
    
    def get_guess(self, raytype):
        if raytype == 1:
            return (self.sg1, self.thetag1)
        if raytype == 2:
            return (self.sg2, self.thetag2)

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
