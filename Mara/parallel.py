
import os
import cPickle
import numpy as np
import cowpy
import h5py
from simulation import MaraEvolutionOperator
from Mara.capis import FluidStateVector


class ParallelSimulation(MaraEvolutionOperator):

    def __init__(self, problem, scheme):

        ng = self.number_guard_zones()
        domain = cowpy.DistributedDomain(problem.resolution, guard=ng)
        descr = problem.fluid_descriptor

        self.shape = tuple([n + 2*ng for n in domain.local_shape])
        self.fluid = FluidStateVector(self.shape, descr)
        self.scheme = scheme
        self.driving = getattr(problem, 'driving', None)
        self.poisson_solver = getattr(problem, 'poisson_solver', None)
        self.domain = domain
        self.problem = problem

        def pad_out(x, v):
            return x + (v,) * (3 - len(self.shape))

        x0 = np.array(problem.lower_bounds)
        x1 = np.array(problem.upper_bounds)
        dX = (x1 - x0) / pad_out(domain.global_shape, 1)
        X0 = x0 + dX * pad_out(domain.global_start, 0)
        X1 = X0 + dX * pad_out(domain.local_shape, 1)

        Nx, Ny, Nz = pad_out(self.fluid.shape, 1)

        dx = (X1[0] - X0[0])/(Nx - (2*ng if Nx > 1 else 0))
        dy = (X1[1] - X0[1])/(Ny - (2*ng if Ny > 1 else 0))
        dz = (X1[2] - X0[2])/(Nz - (2*ng if Nz > 1 else 0))

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.X0, self.X1 = X0, X1

        class ParallelSynchronization(object):
            """
            Currently restricted to periodic BC's
            """
            def __init__(self, mara):
                members = ['rho', 'pre', 'vx', 'vy', 'vz']
                self.domain = mara.domain
                self.fluid = mara.fluid
                self.field = cowpy.DataField(self.domain, members=members)

            def set_boundary(self, X, ng, field='cons'):
                self.field.set_buffer(X)
                self.field.sync_guard()

        self.boundary = ParallelSynchronization(self)

    def number_nonzero(self, X):
        """
        Return the number of nonzero entries of the array X, over all subgrids.
        """
        return self.domain.reduce((X != 0).sum(), type=int, op='sum')

    def write_checkpoint(self, status, dir=".", **extras):
        chkpt_name = "%s/chkpt.%04d.h5" % (dir, status.chkpt_number)
        if self.domain.cart_rank == 0:
            try:
                os.makedirs(dir)
                print "creating data directory", dir
            except OSError: # Directory exists
                pass
            h5f = h5py.File(chkpt_name, 'w')
            chkpt = { "problem": self.problem,
                      "status": status.__dict__ }
            for k,v in chkpt.items():
                if type(v) is dict:
                    grp = h5f.create_group(k)
                    for k1,v1 in v.items():
                        grp[k1] = v1
                else:
                    h5f[k] = cPickle.dumps(v)
            h5f.close()

        field = cowpy.DataField(self.domain, buffer=self.fluid.primitive,
                                members=['rho', 'pre', 'vx', 'vy', 'vz'],
                                name='prim')
        print "Writing checkpoint", chkpt_name
        field.dump(chkpt_name)

    def timestep(self, CFL):
        ml = self.domain.reduce(abs(self.fluid.eigenvalues()).max(), op='max')
        return CFL * self.min_grid_spacing() / ml
