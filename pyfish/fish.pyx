
cimport fish
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free


cdef class FishSolver(object):
    def __cinit__(self):
        self._c = fish_new()

    def __dealloc__(self):
        fish_del(self._c)

    def __init__(self):
        fish_setfluid(self._c, FLUIDS_NRHYD);
        fish_setriemannsolver(self._c, FLUIDS_RIEMANN_EXACT);
        fish_setreconstruction(self._c, FISH_PLM);
        fish_setplmtheta(self._c, 2.0);

    def intercellflux(self, states):
        cdef fluid_state **fluid = <fluid_state**>malloc(
            states.size * sizeof(fluid_state*))
        cdef int i
        cdef FluidState si
        for i in range(states.size):
            si = states[i]
            fluid[i] = si._c
        cdef np.ndarray[np.double_t,ndim=2] Fiph = np.zeros((states.size, 5))
        fish_intercellflux(self._c, fluid, <double*>Fiph.data, states.size, 0)
        free(fluid)
        return Fiph
