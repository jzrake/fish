
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
        pass


cdef class FluidStateVector(object):
    def __cinit__(self, int N):
        self._N = N
        self._c = <fluid_state**> malloc(N * sizeof(fluid_state*))

    def __dealloc__(self):
        free(self._c)

    def __init__(self, shape):
        pass



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
