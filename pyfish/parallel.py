
import os
import numpy as np
import cowpy
from pyfish.simulation import MaraEvolutionOperator
from pyfluids import FluidStateVector


class ParallelSimulation(MaraEvolutionOperator):

    def __init__(self, problem, scheme):
        ng = self.number_guard_zones()
        domain = cowpy.DistributedDomain(problem.resolution, guard=ng)
        descr = problem.fluid_descriptor

        x0 = np.array(problem.lower_bounds)
        x1 = np.array(problem.upper_bounds)
        dX = (x1 - x0) / domain.global_shape
        X0 = x0 + dX * domain.global_start
        X1 = X0 + dX * domain.local_shape

        self.shape = tuple([n + 2*ng for n in domain.local_shape])
        self.fluid = FluidStateVector(self.shape, descr)
        self.scheme = scheme
        self.driving = getattr(problem, 'driving', None)
        self.poisson_solver = getattr(problem, 'poisson_solver', None)
        self.pressure_floor = 1e-6
        self.domain = domain

        Nx, Ny, Nz = self.fluid.shape + (1,) * (3 - len(self.shape))

        dx = (X1[0] - X0[0])/(Nx - (2*ng if Nx > 1 else 0))
        dy = (X1[1] - X0[1])/(Ny - (2*ng if Ny > 1 else 0))
        dz = (X1[2] - X0[2])/(Nz - (2*ng if Nz > 1 else 0))

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.X0, self.X1 = X0, X1

        class ParallelSynchronization(object):
            def __init__(self, mara):
                members = ['rho', 'pre', 'vx', 'vy', 'vz']
                self.domain = mara.domain
                self.fluid = mara.fluid
                self.field = cowpy.DataField(self.domain, members=members)

            def set_boundary(self, X, ng, field='cons'):
                self.field.set_buffer(X)
                self.field.sync_guard()

        self.boundary = ParallelSynchronization(self)

    def write_checkpoint(self, status, dir=".", **extras):
        if self.domain.cart_rank == 0:
            try:
                os.makedirs(dir)
                print "creating data directory", dir
            except OSError: # Directory exists
                pass
        field = cowpy.DataField(self.domain, buffer=self.fluid.primitive,
                                members=['rho', 'pre', 'vx', 'vy', 'vz'],
                                name='prim')
        chkpt_name = "%s/chkpt.%04d.h5" % (dir, status.chkpt_number)
        print "Writing checkpoint", chkpt_name
        field.dump(chkpt_name)
