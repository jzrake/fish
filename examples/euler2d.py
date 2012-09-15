
import time
import numpy as np
import pyfluids
import pyfish

N = 128
dx = 1.0 / N
ng = 3

def set_boundary(U):
    U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
    U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
    U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
    U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]

def dUdt(fluid):
    solver = pyfish.FishSolver()
    solver.reconstruction = "weno5"
    solver.riemannsolver = "exact"
    Fiph = np.zeros([N,N,5])
    Giph = np.zeros([N,N,5])
    L = np.zeros([N,N,5])
    for j in range(N):
        Fiph = solver.intercellflux(fluid._states[:,j], dim=0)
        L[1:,j] += -(Fiph[1:] - Fiph[:-1]) / dx
    for i in range(N):
        Giph = solver.intercellflux(fluid._states[i,:], dim=1)
        L[i,1:] += -(Giph[1:] - Giph[:-1]) / dx
    return L

def advance(fluid, dt):
    U0 = fluid.get_conserved()
    U1 = U0 + 0.5 * dt * dUdt(fluid)
    set_boundary(U1)
    fluid.set_conserved(U1)
    U1 = U0 + 1.0 * dt * dUdt(fluid)
    set_boundary(U1)
    fluid.set_conserved(U1)

t = 0.0
dt = 0.0025

P = np.zeros([N,N,5])
X, Y = np.mgrid[-0.5:0.5:dx,-0.5:0.5:dx]
P[np.where(X**2 + Y**2 >= 0.05)] = [0.1, 0.125, 0.0, 0.0, 0.0]
P[np.where(X**2 + Y**2 <  0.05)] = [1.0, 1.000, 0.0, 0.0, 0.0]

fluid = pyfluids.FluidStateVector([N,N])
fluid.set_primitive(P)

def main():
    iter = 0
    tcur = 0.0
    while tcur < 0.025:
        start = time.clock()
        advance(fluid, dt)
        tcur += dt
        iter += 1
        wall_step = time.clock() - start
        print "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            iter, 0, t, dt, fluid._states.size / wall_step, (wall_step / P.size) * 1e6)

def plot():
    import matplotlib.pyplot as plt
    plt.imshow(fluid.get_primitive()[:,:,0], interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    import cProfile
    cProfile.run('main()')
    #main()
    #plot()
