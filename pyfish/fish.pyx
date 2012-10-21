
cimport pyfish.fish as fish
cimport pyfluids.fluids as fluids
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

def inverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())

_schemes         = {"godunov" : FISH_GODUNOV,
                    "spectral": FISH_SPECTRAL}
_reconstructions = {"pcm"     : FISH_PCM,
                    "plm"     : FISH_PLM,
                    "weno5"   : FISH_WENO5}
_riemannsolvers  = {"hll"     : FLUIDS_RIEMANN_HLL,
                    "hllc"    : FLUIDS_RIEMANN_HLLC,
                    "exact"   : FLUIDS_RIEMANN_EXACT}
_schemes_i = inverse_dict(_schemes)
_reconstructions_i = inverse_dict(_reconstructions)
_riemannsolvers_i = inverse_dict(_riemannsolvers)


cdef class FishSolver(object):
    def __cinit__(self):
        self._c = fish_new()

    def __dealloc__(self):
        fish_del(self._c)

    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)

    def intercellflux(self, states, int dim=0):
        cdef fluids_state **fluid = <fluids_state**>malloc(
            states.size * sizeof(fluids_state*))
        cdef int i
        cdef FluidState si
        for i in range(states.size):
            si = states[i]
            fluid[i] = si._c
        cdef np.ndarray[np.double_t,ndim=2] Fiph = np.zeros([states.size, 5])
        fish_intercellflux(self._c, fluid, <double*>Fiph.data, states.size, dim)
        free(fluid)
        return Fiph

    property scheme:
        def __get__(self):
            cdef int ret
            fish_getparami(self._c, &ret, FISH_SCHEME)
            return _schemes_i[ret]
        def __set__(self, mode):
            fish_setparami(self._c, _schemes[mode], FISH_SCHEME)

    property reconstruction:
        def __get__(self):
            cdef int ret
            fish_getparami(self._c, &ret, FISH_RECONSTRUCTION)
            return _reconstructions_i[ret]
        def __set__(self, mode):
            fish_setparami(self._c, _reconstructions[mode], FISH_RECONSTRUCTION)

    property riemann_solver:
        def __get__(self):
            cdef int ret
            fish_getparami(self._c, &ret, FISH_RIEMANN_SOLVER)
            return _riemannsolvers_i[ret]
        def __set__(self, mode):
            fish_setparami(self._c, _riemannsolvers[mode], FISH_RIEMANN_SOLVER)

    property plm_theta:
        def __get__(self):
            cdef double ret
            fish_getparamd(self._c, &ret, FISH_PLM_THETA)
            return ret
        def __set__(self, plm_theta):
            if not 1.0 <= plm_theta <= 2.0:
                raise ValueError("plm_theta must be between 1 and 2")
            fish_setparamd(self._c, plm_theta, FISH_PLM_THETA)

