
import sys
import os
import time
import pstats
import cProfile
import cPickle
import numpy as np
import pyfluids
import pyfish



class MaraEvolutionOperator(object):
    def __init__(self, problem):
        descr = problem.fluid_descriptor
        X0 = problem.lower_bounds
        X1 = problem.upper_bounds
        ng = self.number_guard_zones()

        self.shape = tuple([n + 2*ng for n in problem.resolution])
        self.fluid = pyfluids.FluidStateVector(self.shape, descr)
        self.solver = pyfish.FishSolver()
        self.solver.reconstruction = "plm"
        self.solver.riemannsolver = "hll"

        if len(self.shape) == 1:
            Nx, Ny, Nz = self.fluid.shape + (1, 1)
        if len(self.shape) == 2:
            Nx, Ny, Nz = self.fluid.shape + (1,)
        if len(self.shape) == 3:
            Nx, Ny, Nz = self.fluid.shape

        dx = (X1[0] - X0[0])/(Nx - (2*ng if Nx > 1 else 0))
        dy = (X1[1] - X0[1])/(Ny - (2*ng if Ny > 1 else 0))
        dz = (X1[2] - X0[2])/(Nz - (2*ng if Nz > 1 else 0))

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.X0, self.X1 = X0, X1

    def set_boundary(self):
        ng = self.number_guard_zones()
        U = self.fluid.conserved()
        self.boundary.set_boundary(self, U)
        self.fluid.from_conserved(U)

    def write_checkpoint(self, status, dir=".", update_status=True, **extras):
        if update_status:
            status.chkpt_last = status.time_current
            status.chkpt_number += 1
        try:
            os.makedirs(dir)
            print "creating data directory", dir
        except OSError: # File exists
            pass
        chkpt = { "prim": self.fluid.primitive, "status": status.__dict__ }
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
        self.boundary.set_boundary(self, U1)
        self.fluid.from_conserved(U1)
        return time.clock() - start

    def dUdt(self, U):
        self.boundary.set_boundary(self, U)
        self.fluid.from_conserved(U)
        L = getattr(self, "_dUdt%dd" % len(self.shape))(self.fluid, self.solver)
        S = self.fluid.source_terms()
        #print S
        return L + S

    def _dUdt1d(self, fluid, solver):
        Nx, = self.fluid.shape
        dx, = 1.0/Nx,
        L = np.zeros([Nx,5])
        Fiph = solver.intercellflux(fluid[:], dim=0)
        L[1:] += -(Fiph[1:] - Fiph[:-1]) / dx
        return L

    def _dUdt2d(self, fluid, solver):
        Nx, Ny = self.fluid.shape
        dx, dy = 1.0/Nx, 1.0/Ny
        L = np.zeros([Nx,Ny,5])
        for j in range(Ny):
            Fiph = solver.intercellflux(fluid[:,j], dim=0)
            L[1:,j] += -(Fiph[1:] - Fiph[:-1]) / dx
        for i in range(Nx):
            Giph = solver.intercellflux(fluid[i,:], dim=1)
            L[i,1:] += -(Giph[1:] - Giph[:-1]) / dy
        return L

    def _dUdt3d(self, fluid, solver):
        Nx, Ny, Nz = self.fluid.shape
        dx, dy, dz = 1.0/Nx, 1.0/Ny, 1.0/Nz
        L = np.zeros([Nx,Ny,Nz,5])
        for j in range(Ny):
            for k in range(Nz):
                Fiph = solver.intercellflux(fluid[:,j,k], dim=0)
                L[1:,j,k] += -(Fiph[1:] - Fiph[:-1]) / dx
        for k in range(Nz):
            for i in range(Nx):
                Giph = solver.intercellflux(fluid[i,:,k], dim=1)
                L[i,1:,k] += -(Giph[1:] - Giph[:-1]) / dy
        for i in range(Nx):
            for j in range(Ny):
                Hiph = solver.intercellflux(fluid[i,j,:], dim=2)
                L[i,j,1:] += -(Hiph[1:] - Hiph[:-1]) / dz
        return L


class SimulationStatus:
    pass


def main():
    interactive_plot = False
    problem = pyfish.problems.OneDimensionalUpsidedownGaussian()
    #problem = pyfish.problems.OneDimensionalPolytrope(tfinal=1.0, fluid='gravp')
    #problem = pyfish.problems.BrioWuShocktube()
    #psolver = pyfish.gravity.PoissonSolver1d()
    mara = MaraEvolutionOperator(problem)
    mara.initial_model(problem.pinit, problem.ginit)
    mara.boundary = problem.build_boundary(mara)

    CFL = 0.4
    chkpt_interval = 1.0

    measlog = { }
    status = SimulationStatus()

    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0

    if interactive_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        lines = plot(mara, None)

    plot(mara, None, show=False)
    while status.time_current < problem.tfinal:
        if interactive_plot:
            lines['rho'].set_ydata(mara.fluid.primitive[:,0])
            lines['pre'].set_ydata(mara.fluid.primitive[:,1])
            lines['vx' ].set_ydata(mara.fluid.primitive[:,2])
            plt.draw()

        ml = abs(mara.fluid.eigenvalues()).max()
        dt = CFL * mara.min_grid_spacing() / ml
        wall_step = mara.advance(dt, rk=3)

        status.time_step = dt
        status.time_current += status.time_step
        status.iteration += 1

        status.message = "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            status.iteration, 0, status.time_current, dt,
            (mara.fluid.size / wall_step) * 1e-3,
            (wall_step / (mara.fluid.size*5)) * 1e6)

        if status.time_current - status.chkpt_last > chkpt_interval:
            mara.write_checkpoint(status, dir="data/test", update_status=True,
                                  measlog=measlog)
            plot(mara, None, show=False, label='%d'%status.iteration)

        measlog[status.iteration] = mara.measure()
        measlog[status.iteration]["time"] = status.time_current
        measlog[status.iteration]["message"] = status.message
        print status.message

    mara.set_boundary()
    return mara, measlog


def plot(mara, measlog, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    if len(mara.shape) == 1:
        #lines['rho'], = plt.plot(mara.fluid.primitive[:,0], '-o', label='density')
        #lines['pre'], = plt.plot(mara.fluid.primitive[:,1], '-o', label='pressure')
        #lines['vx'], = plt.plot(mara.fluid.primitive[:,2], '-o', label='vx')
        if mara.fluid.gravity.size:
            lines['phi'] = plt.plot(mara.fluid.gravity[:,0], '-x', label='phi')
            lines['gph'] = plt.plot(mara.fluid.gravity[:,1], '-x', label='grad phi')
    if len(mara.shape) == 2:
        plt.imshow(mara.fluid.primitive[:,:,0], interpolation='nearest')
    if len(mara.shape) == 3:
        Nx, Ny, Nz = mara.Nx, mara.Ny, mara.Nz
        plot3dslices(mara.fluid.primitive[...,0])
        S, phi = mara.sources.source_terms(mara, retphi=True)
        plot3dslices(phi)
    if show:
        plt.legend()
        plt.show()
    return lines


if __name__ == "__main__":
    #cProfile.run('main()', 'mara_pstats')
    #p = pstats.Stats('mara_pstats')
    #p.sort_stats('time').print_stats()
    mara, measlog = main()
    plot(mara, measlog, label='end')
