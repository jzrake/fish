import pyfish
import pyfluids
import numpy as np

solver = pyfish.FishSolver()
fluid = pyfluids.FluidStateVector([10,10], fluid='gravp')

P = fluid.primitive
P[...] = 1.0
fluid.primitive = P
fluid.gravity[...,0] = 1.0
fluid.gravity[...,1] = 0.0
print fluid.source_terms()

Fiph = solver.intercellflux(fluid[0,:])
print Fiph

#solver.riemannsolver = 'hllc'
#print Fiph

#solver.riemannsolver = 'exact'
#print Fiph
