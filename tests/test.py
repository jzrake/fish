import pyfish
import pyfluids
import numpy as np


solver = pyfish.FishSolver()
vec = pyfluids.FluidStateVector([10,10])

vec._primbuf[0,0,:] = 1.0
print vec._states[0,0].conserved
print type(vec._primbuf)

vec._primbuf[...] = 1.0
Fiph = solver.intercellflux(vec._states[0,:])
print Fiph

# A is a slice, has type state_buffer, but no _gethook attrib
# causes a bug
A = vec._primbuf[0,0]
print A
