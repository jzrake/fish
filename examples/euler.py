
import numpy as np
import pyfluids
import pyfish

N = 100
dx = 1.0 / N
ng = 3

def set_boundary(U):
    U[:+ng] = U[+(ng+0)]
    U[-ng:] = U[-(ng+1)]

def dUdt(fluid):
    solver = pyfish.FishSolver()
    solver.reconstruction = "weno5"
    Fiph = solver.intercellflux(fluid._states)
    L = -(Fiph - np.roll(Fiph, 1, axis=0)) / dx
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

P = np.zeros([N,5])
P[:N/2] = [1.0, 1.000, 0.0, 0.0, 0.0]
P[N/2:] = [0.1, 0.125, 0.0, 0.0, 0.0]

fluid = pyfluids.FluidStateVector([N])
fluid.set_primitive(P)

while t < 0.2:
    advance(fluid, dt)
    t += dt

import matplotlib.pyplot as plt
plt.plot(fluid.get_primitive())
plt.ylim(-0.2, 1.2)
plt.show()
