
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


def test():
    cdef np.ndarray[np.int_t,ndim=2] things = np.zeros((10,10), dtype=np.int)
    cdef int *ptr = <int*>things.data + 2
    cdef int N = 66
    ptr[0] = N
    print things[0,2]

    shape = [10,10]
    states = np.ndarray(shape=shape, dtype=FluidState)
    states[0,1] = FluidState()

    cdef FluidState S = states[0,1]
    cdef np.ndarray[np.double_t,ndim=1] buf = np.zeros(10, dtype=np.double)
    fluids_mapbuffer(S._c, FLUIDS_PRIMITIVE, <double*>buf.data + 0)
    fluids_mapbuffer(S._c, FLUIDS_CONSERVED, <double*>buf.data + 5)

    states[0,1].primitive = np.array([1,1,1,1,1.])
    print states[0,1].primitive
    print states[0,1].conserved
    print buf
