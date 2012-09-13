
import numpy as np
import pyfluids
import pyfish

N = 32
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
    Fiph = np.zeros([N,N,5])
    Giph = np.zeros([N,N,5])
    for j in range(N):
        Fiph[:,j] = solver.intercellflux(fluid._states[:,j], dim=0)
    for i in range(N):
        Giph[i,:] = solver.intercellflux(fluid._states[i,:], dim=1)
        L = -(Fiph - np.roll(Fiph, 1, axis=0)) / dx + \
            -(Giph - np.roll(Giph, 1, axis=1)) / dx
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
dt = 0.001

P = np.zeros([N,N,5])
X, Y = np.mgrid[-0.5:0.5:dx,-0.5:0.5:dx]
P[np.where(X**2 + Y**2 >= 0.05)] = [0.1, 0.125, 0.0, 0.0, 0.0]
P[np.where(X**2 + Y**2 <  0.05)] = [1.0, 1.000, 0.0, 0.0, 0.0]

fluid = pyfluids.FluidStateVector([N,N])
fluid.set_primitive(P)

while t < 0.005:
    print t
    advance(fluid, dt)
    t += dt

import matplotlib.pyplot as plt
plt.imshow(fluid.get_primitive()[:,:,0], interpolation='nearest')
plt.show()
