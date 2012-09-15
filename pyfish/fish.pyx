
cimport pyfish.fish as fish
cimport pyfluids.fluids as fluids
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

def inverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())

_reconstructions = {"none"   : FISH_NONE,
                    "plm"    : FISH_PLM,
                    "weno5"  : FISH_WENO5}
_riemannsolvers  = {"hll"    : FLUIDS_RIEMANN_HLL,
                    "hllc"   : FLUIDS_RIEMANN_HLLC,
                    "exact"  : FLUIDS_RIEMANN_EXACT}
_reconstructions_i = inverse_dict(_reconstructions)
_riemannsolvers_i = inverse_dict(_riemannsolvers)


cdef class FishSolver(object):
    def __cinit__(self):
        self._c = fish_new()

    def __dealloc__(self):
        fish_del(self._c)

    def __init__(self):
        fish_setfluid(self._c, FLUIDS_NRHYD)
        fish_setriemannsolver(self._c, FLUIDS_RIEMANN_EXACT)
        fish_setreconstruction(self._c, FISH_PLM)
        fish_setplmtheta(self._c, 2.0)

    def intercellflux(self, states, int dim=0):
        cdef fluid_state **fluid = <fluid_state**>malloc(
            states.size * sizeof(fluid_state*))
        cdef int i
        cdef FluidState si
        for i in range(states.size):
            si = states[i]
            fluid[i] = si._c
        cdef np.ndarray[np.double_t,ndim=2] Fiph = np.zeros((states.size, 5))
        fish_intercellflux(self._c, fluid, <double*>Fiph.data, states.size, dim)
        free(fluid)
        return Fiph

    property reconstruction:
        def __get__(self):
            cdef int ret
            fish_getreconstruction(self._c, &ret)
            return _reconstructions_i[ret]
        def __set__(self, mode):
            fish_setreconstruction(self._c, _reconstructions[mode])

    property riemannsolver:
        def __get__(self):
            cdef int ret
            fish_getriemannsolver(self._c, &ret)
            return _riemannsolvers_i[ret]
        def __set__(self, mode):
            fish_setriemannsolver(self._c, _riemannsolvers[mode])

    property plm_theta:
        def __get__(self):
            cdef double ret
            fish_getplmtheta(self._c, &ret)
            return ret
        def __set__(self, plm_theta):
            if not 1.0 <= plm_theta <= 2.0:
                raise ValueError("plm_theta must be between 1 and 2")
            fish_setplmtheta(self._c, plm_theta)

