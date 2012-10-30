
import os
import time
import cPickle
import numpy as np
from Mara.capis import FluidStateVector


class MaraEvolutionOperator(object):
    def __init__(self, problem, scheme):
        descr = problem.fluid_descriptor
        X0 = problem.lower_bounds
        X1 = problem.upper_bounds
        ng = self.number_guard_zones()

        self.shape = tuple([n + 2*ng for n in problem.resolution])
        self.fluid = FluidStateVector(self.shape, descr)
        self.scheme = scheme
        self.boundary = problem.build_boundary(self)
        self.driving = getattr(problem, 'driving', None)
        self.poisson_solver = getattr(problem, 'poisson_solver', None)
        self.problem = problem

        Nx, Ny, Nz = self.fluid.shape + (1,) * (3 - len(self.shape))

        dx = (X1[0] - X0[0])/(Nx - (2*ng if Nx > 1 else 0))
        dy = (X1[1] - X0[1])/(Ny - (2*ng if Ny > 1 else 0))
        dz = (X1[2] - X0[2])/(Nz - (2*ng if Nz > 1 else 0))

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.X0, self.X1 = X0, X1

    @property
    def fields(self):
        P = self.fluid.primitive
        G = self.fluid.gravity
        return {'rho': P[...,0],
                'pre': P[...,1],
                'vx' : P[...,2],
                'vy' : P[...,3],
                'vz' : P[...,4],
                'phi': G[...,0] if self.fluid.descriptor.ngravity else None,
                'gph': G[...,1] if self.fluid.descriptor.ngravity else None}

    def write_checkpoint(self, status, dir=".", **extras):
        try:
            os.makedirs(dir)
            print "creating data directory", dir
        except OSError: # Directory exists
            pass
        chkpt = { "prim": self.fluid.primitive,
                  "problem": self.problem,
                  "status": status.__dict__ }
        chkpt.update(extras)
        chkpt_name = "%s/chkpt.%04d.pk" % (dir, status.chkpt_number)
        chkpt_file = open(chkpt_name, 'w')
        print "Writing checkpoint", chkpt_name
        cPickle.dump(chkpt, chkpt_file)

    def measure(self):
        meas = { }
        P = self.fluid.primitive
        U = self.fluid.conserved()
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]
        meas["kinetic"] = (rho * (vx*vx + vy*vy + vz*vz)).mean()
        meas["density_max"] = rho.max()
        meas["density_min"] = rho.min()
        meas["conserved_avg"] = [U[...,i].mean() for i in range(5)]
        meas["primitive_avg"] = [P[...,i].mean() for i in range(5)]
        return meas

    def min_grid_spacing(self):
        return min([self.dx, self.dy, self.dz][:len(self.shape)])

    def number_guard_zones(self):
        return 3

    def number_nonzero(self, X):
        """
        Return the number of nonzero entries of the array X
        """
        return (X != 0).sum()

    def coordinate_grid(self):
        ng = self.number_guard_zones()
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        x0, y0, z0 = self.X0
        x1, y1, z1 = self.X1
        if self.Nx > 1: x0 -= ng * dx
        if self.Ny > 1: y0 -= ng * dy
        if self.Nz > 1: z0 -= ng * dz
        if self.Nx > 1: x1 += ng * dx
        if self.Ny > 1: y1 += ng * dy
        if self.Nz > 1: z1 += ng * dz
        return np.mgrid[x0+dx/2 : x1+dx/2 : dx,
                        y0+dy/2 : y1+dy/2 : dy,
                        z0+dz/2 : z1+dz/2 : dz]

    def initial_model(self, pinit, ginit=None):
        npr = self.fluid.descriptor.nprimitive
        ngr = self.fluid.descriptor.ngravity
        if ginit is None: ginit = lambda x,y,z: np.zeros(ngr)
        shape = self.shape
        X, Y, Z = self.coordinate_grid()
        P = np.ndarray(
            shape=shape + (npr,), buffer=np.array(
                [pinit(x, y, z) for x, y, z in zip(X.flat, Y.flat, Z.flat)]))
        G = np.ndarray(
            shape=shape + (ngr,), buffer=np.array(
                [ginit(x, y, z) for x, y, z in zip(X.flat, Y.flat, Z.flat)]))
        self.fluid.primitive = P
        self.fluid.gravity = G

    def set_boundary(self):
        """
        This function does not call BC's on gravitational field right now.
        """
        ng = self.number_guard_zones()
        self.boundary.set_boundary(self.fluid.primitive, ng, field='prim')

    def advance(self, dt, rk=3):
        start = time.clock()
        U0 = self.fluid.conserved()
        # RungeKuttaSingleStep
        if rk == 1:
            U1 = U0 + dt * self.dUdt(U0)
        # RungeKuttaRk2Tvd
        if rk == 2:
            U1 =      U0 +      dt*self.dUdt(U0)
            U1 = 0.5*(U0 + U1 + dt*self.dUdt(U1))
        # RungeKuttaShuOsherRk3
        if rk == 3:
            U1 =      U0 +                  dt * self.dUdt(U0)
            U1 = 3./4*U0 + 1./4*U1 + 1./4 * dt * self.dUdt(U1)
            U1 = 1./3*U0 + 2./3*U1 + 2./3 * dt * self.dUdt(U1)
        # RungeKuttaClassicRk4
        if rk == 4:
            L1 = self.dUdt(U0)
            L2 = self.dUdt(U0 + (0.5*dt) * L1)
            L3 = self.dUdt(U0 + (0.5*dt) * L2)
            L4 = self.dUdt(U0 + (1.0*dt) * L3)
            U1 = U0 + dt * (L1 + 2.0*L2 + 2.0*L3 + L4) / 6.0

        ng = self.number_guard_zones()
        self.boundary.set_boundary(U1, ng)
        self.from_conserved(U1)

        try:
            if len(self.shape) in [2, 3]:
                self.driving.advance(dt)
                self.driving.drive(self, dt)
        except AttributeError:
            # no driving module
            pass

        return time.clock() - start

    def update_gravity(self):
        """
        Notes:
        ------

        (1) Only works in 1d for now.

        (2) To see the bug introduced by not accounting for the background
        density, do

        self.fluid.descriptor.rhobar = 0.0 #rhobar

        """
        if self.poisson_solver is None:
            return
        if len(self.shape) > 1:
            raise NotImplementedError
        try:
            ng = self.number_guard_zones()
            G, rhobar = self.poisson_solver.solve(self.fields['rho'][ng:-ng],
                                                  retrhobar=True)
            self.fluid.descriptor.rhobar = rhobar
            self.fluid.gravity[ng:-ng] = G
            self.boundary.set_boundary(self.fluid.gravity, ng, field='grav')

        except AttributeError: # no poisson_solver
            pass
        except ValueError: # no gravity array
            pass

    def from_conserved(self, U, safe=True, context=""):
        """
        A safe version of the fluid's from_conserved method.
        """
        if not safe:
            self.fluid.from_conserved(U)
            return
        self.fluid.userflag = 0
        self.fluid.from_conserved(U)
        numerr = self.number_nonzero(self.fluid.userflag)
        if numerr != 0:
            raise RuntimeError("conserved inversion failed on %d zones %s" % (
                    numerr, context))

    def validate_gravity(self):
        if len(self.shape) > 1:
            raise NotImplementedError
        import matplotlib.pyplot as plt
        ng = self.number_guard_zones()
        phi0 = self.fields['phi']
        gph0 = self.fields['gph']
        gph1 = np.gradient(phi0, self.dx)
        plt.semilogy(abs(((gph1 - gph0))[ng:-ng]))
        plt.show()

    def dUdt(self, U):
        ng = self.number_guard_zones()
        dx = [self.dx, self.dy, self.dz]
        self.boundary.set_boundary(U, ng)
        self.from_conserved(U)
        self.update_gravity()
        L = self.scheme.time_derivative(self.fluid, dx)
        if self.fluid.descriptor.fluid in ['gravp', 'gravs']:
            S = self.fluid.source_terms()
            return L + S
        else:
            return L

    def diffusion(self, r):
        """
        Applies diffusion across troubled zones, larger r => more diffusion.
        """
        start = time.clock()
        orig = self.scheme.solver_type
        ng = self.number_guard_zones()
        dx = [self.dx, self.dy, self.dz]
        U = self.fluid.conserved()
        self.boundary.set_boundary(U, ng)
        self.scheme.solver_type = 'diffusion'
        L = self.scheme.time_derivative(self.fluid, dx)
        U += L * r
        self.boundary.set_boundary(U, ng)
        self.from_conserved(U, context="after a diffusion step")
        self.scheme.solver_type = orig
        return time.clock() - start

    def timestep(self, CFL):
        ml = abs(self.fluid.eigenvalues()).max()
        return CFL * self.min_grid_spacing() / ml
