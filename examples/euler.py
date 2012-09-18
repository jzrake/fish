
import time
import cProfile
import pstats
import numpy as np
import pyfluids
import pyfish

AdiabaticGamma = 2.0

class Outflow(object):
    def __init__(self, mara):
        self.mara = mara

    def set_boundary(self, U):
        getattr(self, "set_boundary%dd" % (len(U.shape) - 1))(U)

    def set_boundary1d(self, U):
        ng = self.mara.number_guard_zones()
        U[:+ng] = U[+(ng+0)]
        U[-ng:] = U[-(ng+1)]

    def set_boundary2d(self, U):
        ng = self.mara.number_guard_zones()
        U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
        U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
        U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
        U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]

    def set_boundary3d(self, U):
        ng = self.mara.number_guard_zones()
        U[:,:,:+ng] = U[:,:,+(ng+0)][:,:,np.newaxis,:]
        U[:,:,-ng:] = U[:,:,-(ng+1)][:,:,np.newaxis,:]
        U[:,:+ng,:] = U[:,+(ng+0),:][:,np.newaxis,:,:]
        U[:,-ng:,:] = U[:,-(ng+1),:][:,np.newaxis,:,:]
        U[:+ng,:,:] = U[+(ng+0),:,:][np.newaxis,:,:,:]
        U[-ng:,:,:] = U[-(ng+1),:,:][np.newaxis,:,:,:]


class SelfGravitySourceTerms(object):
    def __init__(self, G=1.0):
        self.G = G

    def get_source_terms(self, mara, retphi=False):
        from numpy.fft import fftfreq, fftn, ifftn
        ng = mara.number_guard_zones()
        G = self.G
        L = 1.0
        Nx, Ny, Nz = mara.fluid._states.shape
        Nx -= 2*ng
        Ny -= 2*ng
        Nz -= 2*ng
        P = mara.fluid.get_primitive()[ng:-ng,ng:-ng,ng:-ng]
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]

        K = [fftfreq(Nx)[:,np.newaxis,np.newaxis] * (2*np.pi*Nx/L),
             fftfreq(Ny)[np.newaxis,:,np.newaxis] * (2*np.pi*Ny/L),
             fftfreq(Nz)[np.newaxis,np.newaxis,:] * (2*np.pi*Nz/L)]
        delsq = -(K[0]**2 + K[1]**2 + K[2]**2)
        delsq[0,0,0] = 1.0 # prevent division by 0

        rhohat = fftn(rho)
        phihat = (4*np.pi*G) * rhohat / delsq
        fx = ifftn(1.j * K[0] * phihat).real
        fy = ifftn(1.j * K[1] * phihat).real
        fz = ifftn(1.j * K[2] * phihat).real
        S = np.zeros(mara.fluid._states.shape + (5,))
        S[ng:-ng,ng:-ng,ng:-ng,0] = 0.0
        S[ng:-ng,ng:-ng,ng:-ng,1] = rho * (fx*vx + fy*vy + fz*vz)
        S[ng:-ng,ng:-ng,ng:-ng,2] = rho * fx
        S[ng:-ng,ng:-ng,ng:-ng,3] = rho * fy
        S[ng:-ng,ng:-ng,ng:-ng,4] = rho * fz
        return (S, ifftn(phihat).real) if retphi else S


class StaticCentralGravity(object):
    def __init__(self, G=1.0, M=1.0):
        self.G = G
        self.M = M

    def get_source_terms(self, mara, retphi=False):
        G = self.G
        M = self.M

        X, Y, Z = mara.coordinate_grid()
        r2 = X**2 + Y**2 + Z**2
        r = r2**0.5

        rhatx, rhaty, rhatz = X/r, Y/r, Z/r
        fx = -G * M * rhatx / r2
        fy = -G * M * rhaty / r2
        fz = -G * M * rhatz / r2

        phi = -G * M / r

        P = mara.fluid.get_primitive()
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]

        S = np.zeros(mara.fluid._states.shape + (5,))
        S[...,0] = 0.0
        S[...,1] = rho * (fx*vx + fy*vy + fz*vz)
        S[...,2] = rho * fx
        S[...,3] = rho * fy
        S[...,4] = rho * fz

        return (S, phi) if retphi else S


class MaraEvolutionOperator(object):
    def __init__(self, shape, X0=[0.0, 0.0, 0.0], X1=[1.0, 1.0, 1.0]):
        self.solver = pyfish.FishSolver()
        self.boundary = Outflow(self)
        self.fluid = pyfluids.FluidStateVector(shape)
        self.sources = None

        self.solver.reconstruction = "none"
        self.solver.riemannsolver = "hll"
        self.shape = tuple(shape)

        for s in self.fluid._states.flat:
            s.gammalawindex = AdiabaticGamma

        if len(shape) == 1:
            Nx, Ny, Nz = self.fluid._states.shape + (1, 1)
        if len(shape) == 2:
            Nx, Ny, Nz = self.fluid._states.shape + (1,)
        if len(shape) == 3:
            Nx, Ny, Nz = self.fluid._states.shape
        dx, dy, dz = (X1[0] - X0[0])/Nx, (X1[1] - X0[1])/Ny, (X1[2] - X0[2])/Nz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.X0, self.X1 = X0, X1

    def measure(self):
        meas = { }
        P = self.fluid.get_primitive()
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]
        meas["kinetic"] = (rho * (vx*vx + vy*vy + vz*vz)).mean()
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

    def initial_model(self, pinit):
        shape = self.shape
        X, Y, Z = self.coordinate_grid()
        P = np.ndarray(
            shape=shape + (5,), buffer=np.array(
                [pinit(x, y, z) for x, y, z in zip(X.flat, Y.flat, Z.flat)]))
        self.fluid.set_primitive(P)

    def advance(self, dt, rk=3):
        U0 = self.fluid.get_conserved()
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
        self.boundary.set_boundary(U1)
        self.fluid.set_conserved(U1)

    def dUdt(self, U):
        assert U.shape[:-1] == self.shape
        self.boundary.set_boundary(U)
        self.fluid.set_conserved(U)
        L = getattr(self, "_dUdt%dd" % len(self.shape))(self.fluid, self.solver)
        if self.sources:
            L += self.sources.get_source_terms(self)
        return L

    def _dUdt1d(self, fluid, solver):
        Nx = self.fluid._states.shape[0]
        dx = 1.0/Nx
        L = np.zeros([Nx,5])
        Fiph = solver.intercellflux(fluid._states, dim=0)
        L[1:] += -(Fiph[1:] - Fiph[:-1]) / dx
        return L

    def _dUdt2d(self, fluid, solver):
        Nx, Ny = self.fluid._states.shape
        dx, dy = 1.0/Nx, 1.0/Ny
        L = np.zeros([Nx,Ny,5])
        for j in range(Ny):
            Fiph = solver.intercellflux(fluid._states[:,j], dim=0)
            L[1:,j] += -(Fiph[1:] - Fiph[:-1]) / dx
        for i in range(Nx):
            Giph = solver.intercellflux(fluid._states[i,:], dim=1)
            L[i,1:] += -(Giph[1:] - Giph[:-1]) / dy
        return L

    def _dUdt3d(self, fluid, solver):
        Nx, Ny, Nz = self.fluid._states.shape
        dx, dy, dz = 1.0/Nx, 1.0/Ny, 1.0/Nz
        L = np.zeros([Nx,Ny,Nz,5])
        for j in range(Ny):
            for k in range(Nz):
                Fiph = solver.intercellflux(fluid._states[:,j,k], dim=0)
                L[1:,j,k] += -(Fiph[1:] - Fiph[:-1]) / dx
        for k in range(Nz):
            for i in range(Nx):
                Giph = solver.intercellflux(fluid._states[i,:,k], dim=1)
                L[i,1:,k] += -(Giph[1:] - Giph[:-1]) / dy
        for i in range(Nx):
            for j in range(Ny):
                Hiph = solver.intercellflux(fluid._states[i,j,:], dim=2)
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


def polytrope2(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-2 # floor (atmospheric) density
    G = 1.0        # gravitational constant
    a = 0.05       # alpha, stellar radius
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
    a = 0.05       # alpha, stellar radius
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 0.5:
        rho = rho_c
    else:
        rho = rho_f
    pre = 1.0
    return [rho, pre, 0.0, 0.0, 0.0]


def main():
    iter = 0
    tcur = 0.0
    CFL = 0.3
    mara = MaraEvolutionOperator([32, 32, 32], X0=[-0.5, -0.5, -0.5], X1=[0.5, 0.5, 0.5])
    #mara.sources = SelfGravitySourceTerms()
    mara.sources = StaticCentralGravity(M=1.0)
    #mara.initial_model(polytrope2)
    mara.initial_model(central_mass)
    measlog = { }
    while tcur < 0.1:
        ml = abs(np.array([f.eigenvalues() for f in mara.fluid._states.flat])).max()
        dt = CFL * mara.min_grid_spacing() / ml
        start = time.clock()
        mara.advance(dt, rk=3)
        tcur += dt
        iter += 1
        wall_step = time.clock() - start
        print "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            iter, 0, tcur, dt,
            (mara.fluid._states.size / wall_step) * 1e-3,
            (wall_step / (mara.fluid._states.size*5)) * 1e6)
        measlog[iter] = mara.measure()
        measlog[iter]["time"] = tcur
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


def plot(mara, measlog):
    import matplotlib.pyplot as plt
    if len(mara.shape) == 1:
        plt.plot(mara.fluid.get_primitive())
    if len(mara.shape) == 2:
        plt.imshow(mara.fluid.get_primitive()[:,:,0], interpolation='nearest')
        plt.show()
    if len(mara.shape) == 3:
        Nx, Ny, Nz = mara.Nx, mara.Ny, mara.Nz
        plot3dslices(mara.fluid.get_primitive()[...,0])
        S, phi = mara.sources.get_source_terms(mara, retphi=True)
        plot3dslices(phi)

    ax = plt.figure().add_subplot(111)
    ax.plot([m["time"] for m in measlog.values()],
            [m["kinetic"] for m in measlog.values()], '-o')
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'mara_pstats')
    #p = pstats.Stats('mara_pstats')
    #p.sort_stats('time').print_stats(10)
    mara, measlog = main()
    plot(mara, measlog)
