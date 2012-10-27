
cimport pyfish.fish as fish
cimport pyfluids.fluids as fluids
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

def inverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())

_solvertypes     = {"godunov"    : FISH_GODUNOV,
                    "spectral"   : FISH_SPECTRAL,
                    "diffusion"  : FISH_DIFFUSION}
_reconstructions = {"pcm"        : FISH_PCM,
                    "plm"        : FISH_PLM,
                    "weno5"      : FISH_WENO5}
_riemannsolvers  = {"hll"        : FLUIDS_RIEMANN_HLL,
                    "hllc"       : FLUIDS_RIEMANN_HLLC,
                    "exact"      : FLUIDS_RIEMANN_EXACT}
_smoothness      = {"jiangshu96" : FISH_ISK_JIANGSHU96,
                    "borges08"   : FISH_ISK_BORGES08,
                    "shenzha10"  : FISH_ISK_SHENZHA10}

_solvertypes_i = inverse_dict(_solvertypes)
_reconstructions_i = inverse_dict(_reconstructions)
_riemannsolvers_i = inverse_dict(_riemannsolvers)
_smoothness_i = inverse_dict(_smoothness)

cdef class FishSolver(object):
    def __cinit__(self):
        self._c = fish_new()

    def __dealloc__(self):
        fish_del(self._c)

    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)

    def intercell_flux(self, states, int dim=0):
        cdef fluids_state **fluid = <fluids_state**>malloc(
            states.size * sizeof(fluids_state*))
        cdef int i
        cdef FluidState si
        cdef int Q = states[0].descriptor.nprimitive
        for i in range(states.size):
            si = states[i]
            fluid[i] = si._c
        cdef np.ndarray[np.double_t,ndim=2] Fiph = np.zeros([states.size, Q])
        fish_intercellflux(self._c, fluid, <double*>Fiph.data, states.size, dim)
        free(fluid)
        return Fiph

    def time_derivative(self, fluidstatevec, spacing):
        states = fluidstatevec.states
        cdef fluids_state **fluid = <fluids_state**>malloc(
            states.size * sizeof(fluids_state*))
        cdef int i, N
        cdef FluidState si
        cdef int Q = fluidstatevec.descriptor.nprimitive
        cdef int shape[3]
        cdef double dx[3]
        for i, N in enumerate(states.shape):
            shape[i] = N
            dx[i] = spacing[i]
        for i in range(states.size):
            si = states.flat[i]
            fluid[i] = si._c
        cdef np.ndarray[np.double_t] L = np.zeros(states.size*Q)
        fish_timederivative(self._c, fluid, len(states.shape), shape, dx,
                            <double*>L.data)
        free(fluid)
        return L.reshape(states.shape + (Q,))

    property solver_type:
        def __get__(self):
            cdef int ret
            fish_getparami(self._c, &ret, FISH_SOLVER_TYPE)
            return _solvertypes_i[ret]
        def __set__(self, mode):
            fish_setparami(self._c, _solvertypes[mode], FISH_SOLVER_TYPE)

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

    property smoothness_indicator:
        def __get__(self):
            cdef int ret
            fish_getparami(self._c, &ret, FISH_SMOOTHNESS_INDICATOR)
            return _smoothness_i[ret]
        def __set__(self, mode):
            fish_setparami(self._c, _smoothness[mode],
                           FISH_SMOOTHNESS_INDICATOR)

    property plm_theta:
        def __get__(self):
            cdef double ret
            fish_getparamd(self._c, &ret, FISH_PLM_THETA)
            return ret
        def __set__(self, plm_theta):
            if not 1.0 <= plm_theta <= 2.0:
                raise ValueError("plm_theta must be between 1 and 2")
            fish_setparamd(self._c, plm_theta, FISH_PLM_THETA)

    property shenzha10_param:
        def __get__(self):
            cdef double ret
            fish_getparamd(self._c, &ret, FISH_SHENZHA10_PARAM)
            return ret
        def __set__(self, plm_theta):
            if not 0.0 <= plm_theta <= 100.0:
                raise ValueError("shenzha10_param must be between 0 and 100")
            fish_setparamd(self._c, plm_theta, FISH_SHENZHA10_PARAM)

