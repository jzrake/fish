
import sys
import os
import time
import pstats
import cProfile
import cPickle
import numpy as np
import pyfluids
import pyfish

AdiabaticGamma = 1.4
FluidSystem = 'gravp'

class Outflow(object):
    def __init__(self):
        pass

    def set_boundary(self, mara, U):
        getattr(self, "set_boundary%dd" % (len(U.shape) - 1))(mara, U)

    def set_boundary1d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:+ng] = U[+(ng+0)]
        U[-ng:] = U[-(ng+1)]

    def set_boundary2d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
        U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
        U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
        U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]

    def set_boundary3d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:,:,:+ng] = U[:,:,+(ng+0)][:,:,np.newaxis,:]
        U[:,:,-ng:] = U[:,:,-(ng+1)][:,:,np.newaxis,:]
        U[:,:+ng,:] = U[:,+(ng+0),:][:,np.newaxis,:,:]
        U[:,-ng:,:] = U[:,-(ng+1),:][:,np.newaxis,:,:]
        U[:+ng,:,:] = U[+(ng+0),:,:][np.newaxis,:,:,:]
        U[-ng:,:,:] = U[-(ng+1),:,:][np.newaxis,:,:,:]


class Inflow(object):
    def __init__(self, SL, SR):
        self.UL = np.array([S.conserved() for S in SL])
        self.UR = np.array([S.conserved() for S in SR])

    def set_boundary(self, mara, U):
        getattr(self, "set_boundary%dd" % (len(U.shape) - 1))(mara, U)

    def set_boundary1d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:+ng] = self.UL
        U[-ng:] = self.UR

    def set_boundary2d(self, mara, U):
        raise NotImplementedError

    def set_boundary3d(self, mara, U):
        raise NotImplementedError


class SafetyModule0(object):
    def __init__(self):
        pass
    def validate(self, fluid, repair=True):
        pass


class SafetyModule1(object):
    def __init__(self):
        pass

    def validate(self, fluid, repair=True):
        P = fluid.primitive
        if (P[...,0] < 0.0).any():
            if repair:
                I = np.where(P[...,1] < 0.0)
                P[...,0][I] = 1e-3
                print "applied density floor to %d zones" % len(I[0])
            else:
                raise RuntimeError("negative density")
        if (P[...,1] < 0.0).any():
            if repair:
                I = np.where(P[...,1] < 0.0)
                P[...,1][I] = 1e-3
                print "applied pressure floor to %d zones" % len(I[0])
            else:
                raise RuntimeError("negative pressure")
        fluid.primitive = P


class MaraEvolutionOperator(object):
    def __init__(self, shape, X0=[0.0, 0.0, 0.0], X1=[1.0, 1.0, 1.0]):
        self.solver = pyfish.FishSolver()
        self.boundary = Outflow()
        self.fluid = pyfluids.FluidStateVector(shape,
                                               fluid=FluidSystem,
                                               gamma=AdiabaticGamma)
        self.sources = None
        self.safety = SafetyModule0()

        self.solver.reconstruction = "plm"
        self.solver.riemannsolver = "hll"
        self.shape = tuple(shape)

        if len(shape) == 1:
            Nx, Ny, Nz = self.fluid.shape + (1, 1)
        if len(shape) == 2:
            Nx, Ny, Nz = self.fluid.shape + (1,)
        if len(shape) == 3:
            Nx, Ny, Nz = self.fluid.shape
        dx, dy, dz = (X1[0] - X0[0])/Nx, (X1[1] - X0[1])/Ny, (X1[2] - X0[2])/Nz
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
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        x0, y0, z0 = self.X0
        x1, y1, z1 = self.X1
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
        if rk == 1:
            """
            RungeKuttaSingleStep
            """
            U1 = U0 + dt * self.dUdt(U0)
        if rk == 2:
            """
            RungeKuttaRk2Tvd
            """
            U1 =      U0 +      dt*self.dUdt(U0)
            U1 = 0.5*(U0 + U1 + dt*self.dUdt(U1))
        if rk == 3:
            """
            RungeKuttaShuOsherRk3
            """
            U1 =      U0 +                  dt * self.dUdt(U0)
            U1 = 3./4*U0 + 1./4*U1 + 1./4 * dt * self.dUdt(U1)
            U1 = 1./3*U0 + 2./3*U1 + 2./3 * dt * self.dUdt(U1)
        if rk == 4:
            """
            RungeKuttaClassicRk4
            """
            L1 = self.dUdt(U0)
            L2 = self.dUdt(U0 + (0.5*dt) * L1)
            L3 = self.dUdt(U0 + (0.5*dt) * L2)
            L4 = self.dUdt(U0 + (1.0*dt) * L3)
            U1 = U0 + dt * (L1 + 2.0*L2 + 2.0*L3 + L4) / 6.0
        self.boundary.set_boundary(self, U1)
        self.fluid.from_conserved(U1)
        self.safety.validate(self.fluid, repair=True)
        return time.clock() - start

    def dUdt(self, U):
        self.boundary.set_boundary(self, U)
        self.fluid.from_conserved(U)
        self.safety.validate(self.fluid, repair=True)
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


def explosion(x, y, z):
    if (x**2 + y**2 + z**2) > 0.05:
        return [0.125, 0.100, 0.0, 0.0, 0.0]
    else:
        return [1.000, 1.000, 0.0, 0.0, 0.0]


def brio_wu(x, y, z):
    if x > 0.0:
        return [0.125, 0.100, 0.0, 0.0, 0.0]
    else:
        return [1.000, 1.000, 0.0, 0.0, 0.0]


def polytrope(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-3 # floor (atmospheric) density
    G = 1.0        # gravitational constant
    b = 0.3        # beta, stellar radius
    a = b / np.pi  # alpha
    n = 1.0        # polytropic index
    K = 4*np.pi*G * a**2 / ((n + 1) * rho_c**(1.0/n - 1.0))
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 1e-6:
        rho = rho_c
    elif r >= np.pi:
        rho = rho_f
    else:
        rho = rho_c * np.sin(r) / r
    pre = K * rho**2
    return [rho, pre, 0.0, 0.0, 0.0]


def central_mass(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-2 # floor (atmospheric) density
    a = 0.3        # alpha, stellar radius
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 0.5:
        rho = rho_c
    else:
        rho = rho_f
    pre = 1.0
    return [rho, pre, 0.0, 0.0, 0.0]


class OneDimensionalUpsidedownGaussian(object):
    sig = 0.05
    sie = 2.00
    ph0 = 1.0
    def ginit(self, x, y, z):
        phi = -self.ph0 * np.exp(-0.5 * x**2 / self.sig**2)
        gph = -x/self.sig**2 * phi
        return [phi, gph, 0.0, 0.0]

    def pinit(self, x, y, z):
        phi = self.ginit(x, y, z)[0]
        e0 = self.sie
        D0 = 1.0
        rho = D0 * np.exp(-phi / (e0 * (AdiabaticGamma - 1.0)))
        pre = rho * e0 * (AdiabaticGamma - 1.0)
        return [rho, pre, 0.0, 0.0, 0.0]


class OneDimensionalPolytrope(object):
    """
    AdiabaticGamma must be 2
    """
    D0 = 1.0
    R = 1.0
    def ginit(self, x, y, z):
        R = self.R
        phi = -(self.D0 / (np.pi/R)**2) * np.cos(np.pi * x / R)
        gph = +(self.D0 / (np.pi/R)**1) * np.sin(np.pi * x / R)
        return [phi, gph, 0.0, 0.0]

    def pinit(self, x, y, z):
        R = self.R
        K = 0.5 * R**2 / np.pi**2
        rho = self.D0 * np.cos(np.pi * x / R)
        pre = K * rho**2.0
        return [rho, pre, 0.0, 0.0, 0.0]


class SimulationStatus:
    pass


def main():
    mara = MaraEvolutionOperator([128], X0=[-0.5,-0.5,-0.5], X1=[0.5,0.5,0.5])
    init = OneDimensionalPolytrope()
    mara.initial_model(init.pinit, init.ginit)
    mara.boundary = Inflow(mara.fluid[0:3], mara.fluid[-3:])

    """
    import matplotlib.pyplot as plt
    phi0 = mara.fluid.gravity[:,0]
    gph0 = mara.fluid.gravity[:,1]
    rho0 = mara.fluid.primitive[:,0]
    pre0 = mara.fluid.primitive[:,1]
    #gph1 = np.gradient(phi0, 1.0/phi0.size)
    lph0 = np.gradient(gph0, 1.0/gph0.size)
    gpre = np.gradient(pre0, 1.0/pre0.size)
    lph1 = mara.fluid.primitive[:,0]

    plt.plot(gpre, label='pressure gradient')
    plt.plot(-rho0 * gph0, label='-rho * dphi/dx')
    #plt.plot(lph1, label='rho')

    plt.legend()
    plt.show()
    exit()
    """

    CFL = 0.2
    chkpt_interval = 2.5 * 2

    measlog = { }
    status = SimulationStatus()

    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0

    plot(mara, None, show=False, label='start')

    while status.time_current < 50:
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


def plot3dslices(A, show=False):
    import matplotlib.pyplot as plt
    Nx, Ny, Nz = A.shape
    ax1 = plt.figure().add_subplot(111)
    ax2 = plt.figure().add_subplot(111)
    ax3 = plt.figure().add_subplot(111)
    ax1.imshow(A[Nx/2,:,:], interpolation='nearest')
    ax2.imshow(A[:,Ny/2,:], interpolation='nearest')
    ax3.imshow(A[:,:,Nz/2], interpolation='nearest')
    if show:
        plt.show()


def plot(mara, measlog, show=True, **kwargs):
    import matplotlib.pyplot as plt
    plt.figure()
    if len(mara.shape) == 1:
        plt.plot(mara.fluid.primitive[:,0], '-o', **kwargs)
        #plt.plot(mara.fluid.gravity[:,0], '-x', **kwargs)
        plt.ylim(0,1)
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


if __name__ == "__main__":
    #cProfile.run('main()', 'mara_pstats')
    #p = pstats.Stats('mara_pstats')
    #p.sort_stats('time').print_stats()
    mara, measlog = main()
    plot(mara, measlog, label='end')
