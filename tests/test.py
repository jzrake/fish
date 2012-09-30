import pyfish
import pyfluids
import numpy as np

descr = pyfluids.FluidDescriptor()
solver = pyfish.FishSolver()
fluid = pyfluids.FluidStateVector([10,10], descr)

P = fluid.get_primitive()
P[...] = 1.0
fluid.set_primitive(P)

Fiph = solver.intercellflux(fluid[0,:])
print Fiph

solver.riemannsolver = 'hllc'
print Fiph

solver.riemannsolver = 'exact'
print Fiph