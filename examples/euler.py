
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
    def __init__(self, problem, scheme):
        descr = problem.fluid_descriptor
        X0 = problem.lower_bounds
        X1 = problem.upper_bounds
        ng = self.number_guard_zones()

        self.shape = tuple([n + 2*ng for n in problem.resolution])
        self.fluid = pyfluids.FluidStateVector(self.shape, descr)
        self.scheme = scheme

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

    @property
    def fields(self):
        P = self.fluid.primitive
        G = self.fluid.gravity
        return {'rho': P[:,0],
                'pre': P[:,1],
                'vx' : P[:,2],
                'vy' : P[:,3],
                'vz' : P[:,4],
                'phi': G[:,0] if self.fluid.descriptor.ngravity else None,
                'gph': G[:,1] if self.fluid.descriptor.ngravity else None}

    def write_checkpoint(self, status, dir=".", update_status=True, **extras):
        if update_status:
            status.chkpt_last = status.time_current
            status.chkpt_number += 1
        try:
            os.makedirs(dir)
            print "creating data directory", dir
        except OSError: # Directory exists
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
        self.boundary.set_boundary(self)
        return time.clock() - start

    def set_boundary(self):
        self.boundary.set_boundary(self)

    def update_gravity(self):
        try:
            ng = self.number_guard_zones()
            G = self.poisson_solver.solve(self.fields['rho'][ng:-ng])
            self.fluid.gravity[ng:-ng] = G
        except AttributeError: # no poisson_solver
            pass
        except ValueError: # no gravity array
            pass

    def validate_gravity(self):
        import matplotlib.pyplot as plt
        phi0 = self.fields['phi']
        gph0 = self.fields['gph']
        gph1 = np.gradient(phi0, self.dx)
        plt.semilogy(abs(((gph1 - gph0))[3:-3]))
        plt.show()

    def dUdt(self, U):
        self.update_gravity()
        self.set_boundary()
        self.fluid.from_conserved(U)
        L = getattr(self, "_dUdt%dd" % len(self.shape))(self.fluid, self.scheme)
        S = self.fluid.source_terms()
        return L + S

    def _dUdt1d(self, fluid, scheme):
        Nx, = self.fluid.shape
        dx, = self.dx,
        L = np.zeros([Nx,5])
        Fiph = scheme.intercellflux(fluid[:], dim=0)
        L[1:] += -(Fiph[1:] - Fiph[:-1]) / dx
        return L

    def _dUdt2d(self, fluid, scheme):
        Nx, Ny = self.fluid.shape
        dx, dy = self.dx, self.dy
        L = np.zeros([Nx,Ny,5])
        for j in range(Ny):
            Fiph = scheme.intercellflux(fluid[:,j], dim=0)
            L[1:,j] += -(Fiph[1:] - Fiph[:-1]) / dx
        for i in range(Nx):
            Giph = scheme.intercellflux(fluid[i,:], dim=1)
            L[i,1:] += -(Giph[1:] - Giph[:-1]) / dy
        return L

    def _dUdt3d(self, fluid, scheme):
        Nx, Ny, Nz = self.fluid.shape
        dx, dy, dz = self.dx, self.dy, self.dz
        L = np.zeros([Nx,Ny,Nz,5])
        for j in range(Ny):
            for k in range(Nz):
                Fiph = scheme.intercellflux(fluid[:,j,k], dim=0)
                L[1:,j,k] += -(Fiph[1:] - Fiph[:-1]) / dx
        for k in range(Nz):
            for i in range(Nx):
                Giph = scheme.intercellflux(fluid[i,:,k], dim=1)
                L[i,1:,k] += -(Giph[1:] - Giph[:-1]) / dy
        for i in range(Nx):
            for j in range(Ny):
                Hiph = scheme.intercellflux(fluid[i,j,:], dim=2)
                L[i,j,1:] += -(Hiph[1:] - Hiph[:-1]) / dz
        return L


class SimulationStatus:
    pass


def main():
    problem_cfg = {'resolution': [128],
                   'tfinal': 5.0,
                   'fluid': 'gravs'}

    problem = pyfish.problems.PeriodicDensityWave(**problem_cfg)
    #problem = pyfish.problems.OneDimensionalUpsidedownGaussian()
    #problem = pyfish.problems.OneDimensionalPolytrope(tfinal=1.0, fluid='gravp')
    #problem = pyfish.problems.BrioWuShocktube()

    scheme = pyfish.FishSolver()
    scheme.reconstruction = "none"
    scheme.riemannsolver = "hllc"

    mara = MaraEvolutionOperator(problem, scheme)
    mara.initial_model(problem.pinit, problem.ginit)
    mara.boundary = problem.build_boundary(mara)
    mara.poisson_solver = pyfish.gravity.PoissonSolver1d()

    mara.update_gravity()
    mara.set_boundary()

    # Status setup
    status = SimulationStatus()
    status.CFL = 0.6
    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0
    status.chkpt_interval = 1.0
    measlog = { }

    # Plotting options
    plot_fields = problem.plot_fields
    plot_interactive = False
    plot_initial = True
    plot_final = True
    problem.plot_fields.append('vx')

    if plot_interactive:
        import matplotlib.pyplot as plt
        plt.ion()
        lines = plot(mara, plot_fields, show=False)

    if plot_initial:
        plot(mara, plot_fields, show=False, label='start')

    while status.time_current < problem.tfinal:
        if plot_interactive:
            for f in plot_fields:
                lines[f].set_ydata(mara.fields[f])
            plt.draw()

        ml = abs(mara.fluid.eigenvalues()).max()
        dt = status.CFL * mara.min_grid_spacing() / ml
        wall_step = mara.advance(dt, rk=3)

        status.time_step = dt
        status.time_current += status.time_step
        status.iteration += 1

        status.message = "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            status.iteration, 0, status.time_current, dt,
            (mara.fluid.size / wall_step) * 1e-3,
            (wall_step / (mara.fluid.size*5)) * 1e6)

        if status.time_current - status.chkpt_last > status.chkpt_interval:
            mara.write_checkpoint(status, dir="data/test", update_status=True,
                                  measlog=measlog)

        measlog[status.iteration] = mara.measure()
        measlog[status.iteration]["time"] = status.time_current
        measlog[status.iteration]["message"] = status.message
        print status.message

    mara.boundary.set_boundary(mara)
    if plot_final:
        plot(mara, plot_fields, show=True, label='end')


def plot(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    try:
        axes = plot.axes
    except:
        plot.axes = [plt.subplot(len(fields),1,n+1) for n,f in enumerate(fields)]
        axes = plot.axes

    for ax, f in zip(axes, fields):
        lines[f], = ax.plot(x.flat, mara.fields[f], '-o', label=(
                f + ' ' + kwargs.get('label', '')))
    if show:
        for ax in axes:
            ax.legend()
        plt.show()
    return lines


if __name__ == "__main__":
    #cProfile.run('main()', 'mara_pstats')
    #p = pstats.Stats('mara_pstats')
    #p.sort_stats('time').print_stats()
    main()
