
import time
import cProfile
import pstats
import numpy as np
import pyfluids
import pyfish


class Outflow(object):
    def __init__(self):
        self.ng = 3

    def set_boundary(self, U):
        getattr(self, "set_boundary%dd" % (len(U.shape) - 1))(U)

    def set_boundary1d(self, U):
        ng = self.ng
        U[:+ng] = U[+(ng+0)]
        U[-ng:] = U[-(ng+1)]

    def set_boundary2d(self, U):
        ng = self.ng
        U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
        U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
        U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
        U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]

    def set_boundary3d(self, U):
        ng = self.ng
        U[:,:,:+ng] = U[:,:,+(ng+0)][:,:,np.newaxis,:]
        U[:,:,-ng:] = U[:,:,-(ng+1)][:,:,np.newaxis,:]
        U[:,:+ng,:] = U[:,+(ng+0),:][:,np.newaxis,:,:]
        U[:,-ng:,:] = U[:,-(ng+1),:][:,np.newaxis,:,:]
        U[:+ng,:,:] = U[+(ng+0),:,:][np.newaxis,:,:,:]
        U[-ng:,:,:] = U[-(ng+1),:,:][np.newaxis,:,:,:]


class MaraEvolutionOperator(object):
    def __init__(self, shape):
        self.solver = pyfish.FishSolver()
        self.boundary = Outflow()
        self.fluid = pyfluids.FluidStateVector(shape)
        self.solver.reconstruction = "weno5"
        self.solver.riemannsolver = "hllc"
        self.shape = tuple(shape)
        if len(shape) == 1:
            Nx, Ny, Nz = self.fluid._states.shape + (1, 1)
            dx, dy, dz = 1.0/Nx, 1.0, 1.0
        if len(shape) == 2:
            Nx, Ny, Nz = self.fluid._states.shape + (1,)
            dx, dy, dz = 1.0/Nx, 1.0/Ny, 1.0
        if len(shape) == 3:
            Nx, Ny, Nz = self.fluid._states.shape
            dx, dy, dz = 1.0/Nx, 1.0/Ny, 1.0/Nz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz

    def min_grid_spacing(self):
        return min([self.dx, self.dy, self.dz][:len(self.shape)])

    def initial_model(self, pinit):
        shape = self.shape
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        X, Y, Z = np.mgrid[-0.5+dx/2:0.5+dx/2:dx,-0.5+dy/2:0.5+dy/2:dy,
                            -0.5+dz/2:0.5+dz/2:dz]
        P = np.array([pinit(x, y, z) for x, y, z in zip(X.flat, Y.flat, Z.flat)])
        self.fluid.set_primitive(P.reshape(shape + (5,)))

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
        return getattr(self, "_dUdt%dd" % len(self.shape))(self.fluid, self.solver)

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


def main():
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
    iter = 0
    tcur = 0.0
    CFL = 0.5
    mara = MaraEvolutionOperator([32, 32, 32])
    mara.initial_model(explosion)
    while tcur < 0.025:
        ml = abs(np.array([f.eigenvalues() for f in mara.fluid._states.flat])).max()
        dt = CFL * mara.min_grid_spacing() / ml
        start = time.clock()
        mara.advance(dt, rk=2)
        tcur += dt
        iter += 1
        wall_step = time.clock() - start
        print "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            iter, 0, tcur, dt,
            (mara.fluid._states.size / wall_step) * 1e-3,
            (wall_step / (mara.fluid._states.size*5)) * 1e6)
    return mara


def plot(mara):
    import matplotlib.pyplot as plt
    if len(mara.shape) == 1:
        plt.plot(mara.fluid.get_primitive())
    if len(mara.shape) == 2:
        plt.imshow(mara.fluid.get_primitive()[:,:,0], interpolation='nearest')
        plt.show()
    if len(mara.shape) == 3:
        Nx, Ny, Nz = mara.Nx, mara.Ny, mara.Nz
        ax1 = plt.figure().add_subplot(111)
        ax2 = plt.figure().add_subplot(111)
        ax3 = plt.figure().add_subplot(111)
        ax1.imshow(mara.fluid.get_primitive()[Nx/2,:,:,0], interpolation='nearest')
        ax2.imshow(mara.fluid.get_primitive()[:,Ny/2,:,0], interpolation='nearest')
        ax3.imshow(mara.fluid.get_primitive()[:,:,Nz/2,0], interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'mara_pstats')
    #p = pstats.Stats('mara_pstats')
    #p.sort_stats('time').print_stats(10)
    mara = main()
    plot(mara)
