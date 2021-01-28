import numpy as np
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult
from scipy.constants import speed_of_light
import importlib
import sys
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Process, Queue

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

    def __init__(self, deriv_folder, deriv_mod, z0, zm, rmax, dr, phi, raytype, label):
        sys.path.append(str(deriv_folder))
        self.dv = importlib.import_module(str(deriv_mod))
        self.label = label
        self.z0 = z0
        self.zm = zm
        self.rmax = rmax
        self.dr = dr
        self.phi = phi
        self.raytype = raytype
        self.dv.phi = self.phi
        self.ray_r = []
        self.ray_z = []
        self.launch_angle = 0.
        self.travel_time = 0.
        self.odesol = []
    
    def copy_ray(self, odesol):
        self.odesol = odesol
        if odesol != (None, None):
            self.launch_angle = odesol.y[0,0]
            self.receive_angle = odesol.y[0,-1]
            self.travel_time = 1e9*odesol.y[2,-1]/speed_of_light
            self.ray_r = np.array(odesol.t)
            self.ray_z = np.array(odesol.y[1,:])

    def comp_ray_parallel(self, q, initguess):
        q.put(self.comp_ray(initguess))


    def comp_ray(self, initguess):
        return self.get_ray_1guess(self.objfn, self.dv.odefns, self.rmax, self.z0, self.zm, self.dr, self.raytype, initguess)

    def hit_top(self, t, y, rt):
        return y[1]

    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800

    def shoot_ray(self, odefn, event, rinit, rmax, theta0, z0, dr, raytype):
        solver = 'RK45'
        #dense_output = True
        sol=solve_ivp(odefn, [rinit, rmax], [theta0, z0, 0], method=solver, events=event, max_step=dr, args=(raytype,))
        if len(sol.t_events[0]) == 0:
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            tinit = sol.t_events[0][0]
            thetainit = sol.y_events[0][0][0]
            travtime = sol.y_events[0][0][2]
            sol2 = solve_ivp(odefn, [tinit, rmax], [np.pi-thetainit, 0, travtime], method=solver, max_step=dr, args=(raytype,))
            tvals = np.hstack((sol.t[sol.t < tinit], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[sol.t < tinit])], sol2.y))
            return OptimizeResult(t=tvals, y=yvals)

    def objfn(self, theta, ode, rmax, z0, zm, dr, raytype):
        #function for rootfinder
        sol = self.shoot_ray(ode, self.hit_top, 0, rmax, theta, z0, dr, raytype)
        zsol = sol.y[1,-1]
        return zsol - zm
    
    def get_ray_1guess(self, minfn, odefn, rmax, z0, zm, dr, raytype, boundguess):
        lb, rb = self.get_bounds_1guess(boundguess, odefn, rmax, z0, zm, dr, raytype)
        if(lb == None and rb == None):
            return None, None
        else:
            minsol = root_scalar(minfn, args=(odefn, rmax, z0, zm, dr, raytype), bracket=[lb,rb])

        print(minsol.converged, minsol.flag)
        odesol = self.shoot_ray(odefn, self.hit_top, 0, rmax, minsol.root, z0, dr, raytype)
        '''
        self.launch_angle = odesol.y[0,0]
        self.receive_angle = odesol.y[0,-1]
        self.travel_time = 1e9*odesol.y[2,-1]/speed_of_light
        self.ray_r = np.array(odesol.t)
        self.ray_z = np.array(odesol.y[1,:])
        '''
        return odesol

    def get_bounds_1guess(self, initguess, odefn, rmax, z0, zm, dr, raytype, odeparam = 'r', xtol = None, maxiter=None):
        if xtol != None:
            xtol = xtol
        else:
            xtol=1e-4

        if maxiter != None:
            maxiter = maxiter
        else:
            maxiter=200

        param = odeparam

        dxi=1e-2
        dx = dxi
        zendintl = self.objfn(initguess, odefn, rmax, z0, zm, dr, raytype)
        dz = np.sign(zendintl)
        grad = np.sign(self.objfn(initguess+dx, odefn, rmax, z0, zm, dr, raytype) - zendintl)
        print(dz, grad)
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
            zendnew = self.objfn(newguess, odefn, rmax, z0, zm, dr, raytype)

        if newguess >= np.pi/2 or newguess <= 0:
            print('ERROR: no interval found')
            print(np.sign(zendintl))
            return None, None
        else:
            lb, rb = lastguess, newguess

        print('initial guess:', initguess, 'returned bounds:', lb, rb)
        return lb, rb
