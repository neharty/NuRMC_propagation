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

    def __init__(self, z0, zf, rf, dr, phi, eps, ntype, raytype, label = None):
        self.label = label
        self.z0 = z0
        self.zf = zf
        self.rf = rf
        self.dr = dr
        self.phi = phi
        self.ntype = ntype
        self.raytype = raytype
        self.phi = self.phi
        self.eps = eps
        self.ray_r = []
        self.ray_z = []
        self.ray = OptimizeResult(t=[], y=[])
        self.launch_angle = 0.
        self.travel_time = 0.
        self.odesol = []
    
    def copy_ray(self, odesol):
        self.ray = odesol
        if odesol != (None, None):
            self.launch_angle = odesol.y[0,0]
            self.receive_angle = odesol.y[0,-1]
            self.travel_time = 1e9*odesol.y[2,-1]/speed_of_light
            self.ray_r = np.array(odesol.t)
            self.ray_z = np.array(odesol.y[1,:])

    def comp_ray_parallel(self, q):
        q.put(self.get_ray())
 
    def adj(self, A):
        #formula for a 3x3 matrix
        return 0.5*np.eye(3)*(np.trace(A)**2 - np.trace(A@A)) - np.trace(A)*A + A@A

    def n(self, z, phi, theta):
        tmp = self.eps(z)
        if not isinstance(tmp, np.ndarray):
            return np.sqrt(self.eps(z))
        else:
            rdot = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
            A = rdot @ self.eps(z) @ rdot
            B = rdot @ (np.trace(self.adj(self.eps(z)))*np.eye(3) - self.adj(self.eps(z))) @ rdot
            C = np.linalg.det(self.eps(z))
            A, B, C = np.longdouble(A), np.longdouble(B), np.longdouble(C)
            discr = (B + 2*np.sqrt(A*C))*(B - 2*np.sqrt(A*C))
            ntmp = np.sqrt((B + np.sqrt(np.abs(discr)))/(2*A))

            no = np.sqrt(self.eps(z)[0,0])
            ne = np.sqrt(self.eps(z)[2,2])

            if self.ntype == 1:
                return no*ne/np.sqrt(ne**2*np.cos(theta)**2+no**2*np.sin(theta)**2)
                #return np.sqrt(C/A)/ntmp
            if self.ntype == 2:
                #print(np.abs(ntmp - no))
                #return ntmp
                return no
    
    def rayode(self, t, y):
        n = self.n(y[1], self.phi, y[0])
        #alpha = np.cos(y[0])
        return [-np.cos(y[0])*self.zderiv(y[1], self.phi, y[0])/(n*np.cos(y[0])+self.thetaderiv(y[1], self.phi, y[0])*np.sin(y[0])), 1/np.tan(y[0]), n/np.abs(np.sin(y[0]))]
    
    def zderiv(self, z, phi, theta, h=1e-5):
        return (self.n(z + h, phi, theta) - self.n(z-h, phi, theta))/(2*h)

    def thetaderiv(self, z, phi, theta, h=1e-5):
        return (self.n(z, phi, theta + h) - self.n(z, phi, theta - h))/(2*h)

    def hit_top(self, t, y):
        return y[1]

    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800

    def shoot_ray(self, theta0):
        solver = 'RK45'
        #dense_output = True
        sol=solve_ivp(self.rayode, [0, self.rf], [theta0, self.z0, 0], method=solver, events=self.hit_top, max_step=self.dr)
        if len(sol.t_events[0]) == 0:
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            tinit = sol.t_events[0][0]
            thetainit = sol.y_events[0][0][0]
            travtime = sol.y_events[0][0][2]
            sol2 = solve_ivp(self.rayode, [tinit, self.rf], [np.pi-thetainit, 0, travtime], method=solver, max_step=self.dr)
            tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
            return OptimizeResult(t=tvals, y=yvals)

    def objfn(self, theta):
        #function for rootfinder
        sol = self.shoot_ray(theta)
        zsol = sol.y[1,-1]
        return zsol - self.zf
    
    def _get_ray(self):
        lb, rb = self.get_bounds()
        if(lb == None and rb == None):
            return None, None
        else:
            minsol = root_scalar(self.objfn, bracket=[lb,rb])

        print(minsol.converged, minsol.flag)
        odesol = self.shoot_ray(minsol.root)
        return odesol
    
    def get_ray(self):
        if self.ray.t == []:
            self.ray = self._get_ray()
            print(self.zf, self.ray.y[1, -1])
            return self.ray
        else:
            return self.ray

    def get_bounds(self, xtol = None, maxiter=None):
        initguess = self.get_guess()
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
        zendintl = self.objfn(initguess)
        dz = np.sign(zendintl)
        grad = np.sign(self.objfn(initguess+dx) - zendintl)
        #print(dz, grad)
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
            zendnew = self.objfn(newguess)

        if newguess >= np.pi/2 or newguess <= 0:
            print('ERROR: no interval found')
            #print(np.sign(zendintl))
            return None, None
        else:
            lb, rb = lastguess, newguess

        print('initial guess:', initguess, 'returned bounds:', lb, rb)
        return lb, rb

    def get_guess(self, q=None):
        r0 = np.array([0, 0, self.z0])
        rf = np.array([self.rf, 0, self.zf])
        g = analyticraytracing.ray_tracing(r0, rf, medium.get_ice_model('ARAsim_southpole'), n_frequencies_integration = 1)
        g.find_solutions()
        lv = g.get_launch_vector(self.raytype-1)
        lv = lv/np.linalg.norm(lv)
        thetag = np.arccos(lv[2])
        return thetag

