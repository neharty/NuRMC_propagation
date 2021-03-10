import numpy as np
import autograd.numpy as anp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult, least_squares
from scipy.optimize import differential_evolution
from scipy.constants import speed_of_light
from scipy.linalg import null_space
from multiprocessing import Process, Queue
from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.utilities import medium
from autograd import grad

class ray:
    '''
    utility class for 1 ray

    rays are defined by start and end points, type of 'index of refraction' used, and the type of ray (direct of refracted/reflected)

    object attributes:
        * x0, y0, x0 : initial ray coordinates
        * xf, yf, zf : final ray coordinates
        * ray : solution of ODE integrator
            ray.t : discrete arclength values along the ray
            ray.y : discrete ray position, momentum and (unscaled) time values along the ray
        * travel_time : final travel time for ray, in nanoseconds
        * launch_vector : unit vector for the ray's launch direction
        * receive_vector : unit vector for the ray's receive direction 
        * initial_wavefront : unit vector for the initial p vector
        * final_wavefront : unit vector for the final p vector
        * initial/final_E_pol : unit vector for the initial/final electric field polarization
        * initial/final_B_pol : unit vector for the intial/final magnetic flux density polarization
            NOTE: B and H are parallel since the permeability = 1, generally this is not true
    '''

    def __init__(self, x0, y0, z0, xf, yf, zf, ntype, raytype, eps, label=None):
        '''
        initializes object
        
        x0, y0, z0 : intial cartesian coordinates of the ray

        xf, yf, zf : final cartesian coordinates fo the ray

        ntype : selects which root to use for |p| calculation, can be either 1 or 2
            * in uniaxial media, 1 = extraordinary and 2 = ordinary root

        raytype : defines which solution to search for, can be either 1 or 2
            * 1 = directish, in NRMC this is the direct ray
            * 2 = refractedish, in NRMC this can be either the reflected or refracted ray

        eps : function handle for the material's permittivity, must be a function of position only
              if you want to use NRMC raytracing for initial guesses, it must be an exponential profile for the scalar index of refraction approximation
        '''

        self.label = label

        self.x0, self.y0, self.z0 = x0, y0, z0
        self.xf, self.yf, self.zf = xf, yf, zf
        
        if raytype == 1 or raytype == 2:
            self.raytype = raytype
        else:
            print(raytype)
            raise RuntimeError('Please enter a valid ray type (1 or 2)')
        
        if ntype == 1 or ntype == 2:
            self.ntype = ntype
        else:
            print(ntype)
            raise RuntimeError('Please enter a valid index of refraction type (1 or 2)')
        
        self.eps = eps

        self.ray_x = []
        self.ray_y = []
        self.ray_z = []
        self.travel_time = 0.
        self.ray = OptimizeResult(t=[], y=[])
        self.launch_vector = []
        self.receive_vector = []
        self.initial_E_pol = []
        self.initial_B_pol = []
        self.final_E_pol = []
        self.final_B_pol = []
        self._dsign = 1

    def copy_ray(self, odesol):
        if odesol != (None, None):
            self.ray = odesol
            self.travel_time = 1e9*odesol.y[-1,-1]/speed_of_light
            self.launch_vector = self._unitvect(np.array(self.ray.y[0:3, 1] - self.ray.y[0:3, 0]))
            self.receive_vector = self._unitvect(np.array(self.ray.y[0:3, -1] - self.ray.y[0:3,-2]))
            self.initial_wavefront = self._unitvect(np.array(self.ray.y[3:6, 0]))
            self.final_wavefront = self._unitvect(np.array(self.ray.y[3:6, -1]))

            self.initial_E_pol, self.initial_B_pol = self._get_pols(0)
            self.final_E_pol, self.final_B_pol = self._get_pols(-1)

    def get_ray_parallel(self, q, sguess, phiguess, thetaguess, thetabound):
        q.put(self.get_ray(sguess, phiguess, thetaguess, thetabound)) 

    def hit_top(self, s, u):
        return u[2]
    
    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800
    
    def ode(self, s, u):
        '''
        RHS of q and p ODE system, with travel time ODE

        takes in values and converts derivatives w.r.t. arclength
        '''
        
        q = u[:3]
        p = u[3:-1]
        phat = p/np.linalg.norm(p, 2)
        qdot = self.DpH(q, p)
        
        qdn = np.linalg.norm(qdot, 2)
        
        qdot = self._dsign*qdot/qdn
        pdot = -self._dsign*self.DqH(q, p)/qdn
        
        if np.dot(phat, qdot) > 1:
            cosang = 1
        elif np.dot(phat, qdot) < -1: 
            cosang = -1
        else:
            cosang = np.dot(phat, qdot)

        return np.array([qdot[0], qdot[1], qdot[2], pdot[0], pdot[1], pdot[2], np.dot(p, qdot)])
   
    def HMat(self, q, p):
        D2 = np.array([[-p[1]**2 - p[2]**2, p[0]*p[1], p[0]*p[2]],
            [p[0]*p[1], -p[0]**2-p[2]**2, p[1]*p[2]],
            [p[2]*p[0], p[2]*p[1], -p[0]**2-p[1]**2]])
        return self.eps(q)+D2
    
    def det(self, A):
        #return A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[0,1]*A[2,2] - A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
        return np.linalg.det(A)

    def H(self, q, p):
        return np.linalg.det(self.HMat(q, p))

    def DqH(self, q, p):
        q = np.array(q)
        p = np.array(p)
        macheps = 7./3 - 4./3 -1
        h = (macheps)**(2/3)*np.max(np.abs(q))
        DxH = (self.H([q[0] + h, q[1], q[2]], p) - self.H([q[0] - h, q[1], q[2]], p))/(2*h)
        DyH = (self.H([q[0], q[1] + h, q[2]], p) - self.H([q[0], q[1]- h, q[2]], p))/(2*h)
        DzH = (self.H([q[0], q[1], q[2] + h], p) - self.H([q[0], q[1], q[2] - h], p))/(2*h)
        return np.array([DxH, DyH, DzH])

    def DpH(self, q, p):
        q = np.array(q)
        p = np.array(p)
        macheps = 7./3 - 4./3 -1
        h = (macheps)**(2/3)*np.max(np.abs(p))
        Dp1H = (self.H(q, [p[0] + h, p[1], p[2]]) - self.H(q, [p[0] - h, p[1], p[2]]))/(2*h)
        Dp2H = (self.H(q, [p[0], p[1] + h, p[2]]) - self.H(q, [p[0], p[1] - h, p[2]]))/(2*h)
        Dp3H = (self.H(q, [p[0], p[1], p[2] + h]) - self.H(q, [p[0], p[1], p[2] - h]))/(2*h)
        return np.array([Dp1H, Dp2H, Dp3H])
        
    def adj(self, A):
        '''computes the adjugate for a 3x3 matrix'''
        return 0.5*np.eye(3)*(np.trace(A)**2 - np.trace(A@A)) - np.trace(A)*A + A@A
    
    def n(self, r, rdot):
        '''
        computes |p| = n using formula given in Chen

        r : position
        rdot : p direction
        '''
        tmp = self.eps(r)
        if not isinstance(tmp, np.ndarray):
            return np.sqrt(tmp)
        else:
            rdot = rdot/np.linalg.norm(rdot)
            A = rdot @ self.eps(r) @ rdot
            B = rdot @ (np.trace(self.adj(self.eps(r)))*np.eye(3) - self.adj(self.eps(r))) @ rdot
            C = np.linalg.det(self.eps(r))
            discr = (B + 2*np.sqrt(A*C))*(B - 2*np.sqrt(A*C))
            ntmp = (B + np.sqrt(np.abs(discr)))/(2*A)
            
            if self.ntype == 1:
                return np.sqrt(C/(ntmp*A))
            if self.ntype == 2:
                return np.sqrt(ntmp)

    def shoot_ray(self, sf, phi0, theta0): 
        '''
        solves the ray ODEs given arclength and intial angles for p

        sf : final arclength value for integration
        phi0 : azimuth angle for p
        theta0 : zenith angle for p
        '''
        
        idir = np.array([np.cos(phi0)*np.sin(theta0), np.sin(phi0)*np.sin(theta0), np.cos(theta0)])
        dx0, dy0, dz0 = self.n([self.x0, self.y0, self.z0], idir)*idir
        #dot = np.dot(idir, self._unitvect(self.DpH([self.x0, self.y0, self.z0], [dx0, dy0, dz0])))
        if 1 == np.sign(self.DpH([self.x0, self.y0, self.z0], [dx0, dy0, dz0])[-1]):
            self._dsign = 1
        else:
            self._dsign = -1

        solver = 'RK45'
        mstep = max(np.abs(sf), np.sqrt((self.xf - self.x0)**2+(self.yf-self.y0)**2 + (self.zf-self.z0)**2))/30
        sol=solve_ivp(self.ode, [0, sf], [self.x0, self.y0, self.z0, dx0, dy0, dz0, 0], method=solver, events=self.hit_top, max_step=mstep) 
        if len(sol.t_events[0]) == 0:
            sol.t = np.abs(sol.t)
            sol.y[-1,:] = np.abs(sol.y[-1,:])
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            sinit = sol.t_events[0][0]
            evnt = sol.y_events[0][0]
            sol2 = solve_ivp(self.ode, [sinit, sf], [evnt[0], evnt[1], 0, evnt[3], evnt[4], -evnt[5], evnt[6]], method=solver, max_step=mstep)
            tvals = np.hstack((sol.t[:-1], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[:-1])], sol2.y))
            tvals = np.abs(tvals)
            yvals[-1,:] = np.abs(yvals[-1,:])
            return OptimizeResult(t=tvals, y=yvals)

    def _rootfn(self, args):
        '''
        function for rootfinder, returns absolute distance of x, y, z components
        
        args: tuple of arclength, azimuth and zenith angles
        '''
        sol = self.shoot_ray(args[0], args[1], args[2])
        return np.abs([sol.y[0, -1] - self.xf, sol.y[1, -1] - self.yf, sol.y[2, -1] - self.zf])

    def _distsq(self, args):
        '''
        function for rootfinder, returns absolute distance (scalar)
        
        args: tuple of arclength, azimuth and zenith angles
        '''
        sol = self.shoot_ray(args[0], args[1], args[2])
        return (sol.y[0, -1] - self.xf)**2 +  (sol.y[1, -1] - self.yf)**2 + (sol.y[2, -1] - self.zf)**2
    
    def _unitvect(self, v):
        '''returns unit vector in the direction of v'''
        return v/np.linalg.norm(v)
    
    def _get_pols(self, idx):
        '''
        given an array index idx, computes E and B fields at position[idx] and p[idx] along the ray using formulas in Chen
        
        idx : array index
        '''
        if True:
            return [0,0,0], [0,0,0]

        q = self.ray.y[0:3, idx]
        p = self.ray.y[3:6, idx]
        n = np.linalg.norm(p)
        
        det = lambda a : np.linalg.det(a)
        
        e = [0,0,0]
        b = [0,0,0]

        J = self.eps(q) - n**2*np.eye(3)
        J[np.abs(J) < 1e-5] = 0
        dJ = np.abs(det(J))
            
        print('\n-------------')

        if dJ > 1e-10:
            print('J is nonsingular')
            e = self.adj(J) @ p
            e = e / np.linalg.norm(e)
            b = np.cross(p, e)
            b = b / np.linalg.norm(b)
        else:
            print('J is singular')
            ns = null_space(J)
            #nzc = np.nonzero(np.any(adjJ != 0, axis=0))[0][0]
            dots = np.dot(ns.T, p/np.linalg.norm(p))
            u = ns[:, np.argmin(dots)]
            b = np.cross(p, u)
            b = b / np.linalg.norm(b)
            e = -np.linalg.inv(self.eps(q)) @ b
            e = e / np.linalg.norm(e)

            '''
            testm = np.abs(self.adj(J))
            testm = testm > 1e-10

            bl = True
            for row in testm:
                for col in row:
                    if col == False:
                        bl = False
            
            adjJ = self.adj(J)
            adjJ[np.abs(adjJ) < 1e-10] = 0
            
            print(J)
            print(adjJ)
            
            if bl:
                print('J is singular, adj(J) =/= 0')
                nzc = np.nonzero(np.any(adjJ != 0, axis=0))[0][0]
                e = adjJ[:,nzc].T
                e = e / np.linalg.norm(e)
            else:
                print('J is singular, adj(J) == 0')
                nzc = np.nonzero(np.any(J != 0, axis=0))[0][0]
                e = J[:,nzc].T
                e = e / np.linalg.norm(e)
            '''
        b = np.cross(p, e)
        b = b / np.linalg.norm(b)

        print(self.ntype, self.raytype, e, b)
        print(np.dot(p, self.eps(q) @ e), np.dot(p, b))
        return e, b

    def _get_ray(self, sf, phi, theta, thetabound):
        '''
        finds ray using rootfinding via the shooting method for BVPs, then computes ray attributes

        sf : initial guess for ray's arclength
        phi : intial guess for azimuth angle
        theta : intial guess for zenith angle
        '''
        
        def distsqarclen(s):
            return self._distsq((s, phi, theta))

        if self.ray.t == []:
            #minsol = root(self._rootfn, [sf, phi, theta], options={'xtol': 1e-10, 'eps':1e-3, 'factor': 1, 'diag': None})
            #minsol = least_squares(self._rootfn, [sf, phi, theta], method='trf', bounds=([-np.inf, 0, 0], [np.inf,  2*np.pi, thetabound]), xtol=1e-10, x_scale=[1000, 1e-8, 0.1])
            #minphi = root(rootfn1, phi, method='lm')
            arclen = root(distsqarclen, sf, method='lm', options={'ftol':1e-10, 'xtol':1e-8, 'maxiter':500, 'eps':1e-8})
            #self.copy_ray(self.shoot_ray(arclen.x, phi, theta))
            minsol = least_squares(self._rootfn, [arclen.x[0], phi, theta], method='lm', ftol=1e-10, xtol=1e-10, x_scale=[1000, 1e-8, 1e-2], verbose=1)
            ms = minsol.x
            if minsol.cost > 1e-3:
                #minsol = minimize(self._distsq, minsol.x, method='Powell', options={'disp':True, 'maxiter':1000, 'xtol':1e-6, 'ftol':1e-6})
                #minsol = minimize(self._distsq, minsol.x, method='Nelder-Mead', options={'disp':True, 'maxiter':1000, 'xtol':1e-6, 'ftol':1e-6, 'adaptive':True})
                #minsol = minimize(self._distsq, minsol.x, method='BFGS', options={'maxiter':1000, 'gtol':1e-10, 'norm':2})
                #minsol = root(self._rootfn, minsol.x, options={'xtol':1e-10, 'factor':10, 'diag':[1000, 1e-8, 0.01]})
                #minsol = root(self._rootfn, minsol.x, method='lm', options={'ftol':1e-12, 'xtol':1e-12, 'maxiter':500, 'eps':1e-8, 'diag':(minsol.x[0], 1e-10, minsol.x[2])})
                ms = self.secant(self._distsq, ms)
                #print([sf, phi, theta], minsol.x)
            #print(self.ntype, self.raytype, minsol.success, minsol.message)
            
            self.copy_ray(self.shoot_ray(*ms))
            #self.copy_ray(self.shoot_ray(sf, phi, theta))

            return self.ray
        else:
            return self.ray
    def secant(self, f, x0, ftol=1e-6, maxiter=200):
        
        macheps = 7./3 - 4./3 - 1
        h = np.sqrt(macheps)
        h = 1e-4
        def jac(x):
            return (f(x + h) - f(x))/h
        x0 = np.array(x0) 
        xk = x0
        fk = f(x0)
        xk1 = xk+[x0[0] + np.sqrt(fk), 0, h]
        fk1 = f(xk1)
        xk11 = xk1+[xk1[0] + np.sqrt(fk1), 0, h]
        fk = f(x0)
        fk1 = f(xk1)

        iternum = 0
        while np.abs(f(xk1)) > ftol and iternum < maxiter:
            xtmp = xk11
            xk11 = xk1-(fk1/(fk1-fk))*(xk1 - xk)
            xk = xk1
            xk1 = xtmp
            fk = f(xk)
            fk1 = f(xk1)
            iternum = iternum + 1
        print(fk1, iternum)
        return xk11
            

    def get_ray(self, sg, phig, thetag, thetabound):
        if self.ray.t == []:
            #sg, phig, thetag = self.get_guess()            
            self._get_ray(sg, phig, thetag, thetabound)
            print(self.ntype, self.raytype, self.xf, self.yf, self.zf, self.ray.y[0, -1], self.ray.y[1,-1], self.ray.y[2,-1])
            return self.ray
        else:
            return self.ray
        
class rays(ray):
    '''
    wrapper for ray class

    finds all 4 solutions: 1 refracted and 1 direct for each of the 2 ntype calculations
    '''

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
        g = analyticraytracing.ray_tracing(np.array([self.x0, self.y0, self.z0]), np.array([self.xf, self.yf, self.zf]), medium.get_ice_model('ARAsim_southpole'), n_frequencies_integration = 1)
        g.find_solutions()
        self.sg1, self.sg2 = g.get_path_length(0), g.get_path_length(1)
                
        lv1, lv2 = g.get_launch_vector(0), g.get_launch_vector(1)
        lv1, lv2 = lv1/np.linalg.norm(lv1), lv2/np.linalg.norm(lv2)
        
        self.phig = np.arctan2((self.yf - self.y0), (self.xf-self.x0))
        
        self.thetag1, self.thetag2 = np.arccos(lv1[2]), np.arccos(lv2[2])
    
    def get_guess(self, raytype):
        if raytype == 1:
            return (self.sg1, self.phig, self.thetag1, np.pi)
        if raytype == 2:
            return (self.sg2, self.phig, self.thetag2, self.thetag1)

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
            
            for i in [0,1]:
                if max(self.r[i,0].ray.y[2,:]) > max(self.r[i,1].ray.y[2,:]):
                    tmp = self.r[i,0]
                    self.r[i,0] = self.r[i,1]
                    self.r[i,1] = tmp
                    del tmp
            
    def get_ray(self, i, k):
        return self.r[i,k].ray

    def get_time(self, i, k):
        return self.r[i,k].travel_time
    
    def get_initial_E_pol(self, i, k):
        return self.r[i,k].initial_E_pol

    def get_final_E_pol(self, i, k):
        return self.r[i,k].final_E_pol

    def get_initial_B_pol(self, i, k):
        return self.r[i,k].initial_B_pol

    def get_final_B_pol(self, i, k):
        return self.r[i,k].final_B_pol
