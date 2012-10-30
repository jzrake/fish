
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport capis
import numpy as np


def inverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())

_solvertypes     = {"godunov"    : FISH_GODUNOV,
                    "spectral"   : FISH_SPECTRAL,
                    "diffusion"  : FISH_DIFFUSION}
_reconstructions = {"pcm"        : FISH_PCM,
                    "plm"        : FISH_PLM,
                    "weno5"      : FISH_WENO5}
_smoothness      = {"jiangshu96" : FISH_ISK_JIANGSHU96,
                    "borges08"   : FISH_ISK_BORGES08,
                    "shenzha10"  : FISH_ISK_SHENZHA10}
_riemannsolvers  = {"hll"        : FLUIDS_RIEMANN_HLL,
                    "hllc"       : FLUIDS_RIEMANN_HLLC,
                    "exact"      : FLUIDS_RIEMANN_EXACT}

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

    property options:
        def __get__(self):
            return {'solver_type': self.solver_type,
                    'reconstruction': self.reconstruction,
                    'riemann_solver': self.riemann_solver,
                    'shenzha10_param': self.shenzha10_param,
                    'smoothness_indicator': self.smoothness_indicator }

    def __repr__(self):
        props = ["%s" % (type(self)),
                 'solver_type: %s' % self.solver_type,
                 'reconstruction: %s' % self.reconstruction,
                 'riemann_solver: %s' % self.riemann_solver,
                 'shenzha10_param: %f' % self.shenzha10_param,
                 'smoothness_indicator: %s' % self.smoothness_indicator]
        return "{" + "\n\t".join(props) + "}"


_fluidsystem = {"nrhyd"         : FLUIDS_NRHYD,
                "gravs"         : FLUIDS_GRAVS,
                "gravp"         : FLUIDS_GRAVP,
                "srhyd"         : FLUIDS_SRHYD}
_coordsystem = {"cartesian"     : FLUIDS_COORD_CARTESIAN,
                "spherical"     : FLUIDS_COORD_SPHERICAL,
                "cylindrical"   : FLUIDS_COORD_CYLINDRICAL}
_equationofstate = {"gammalaw"  : FLUIDS_EOS_GAMMALAW,
                    "tabulated" : FLUIDS_EOS_TABULATED}
_riemannsolver = {"hll"         : FLUIDS_RIEMANN_HLL,
                  "hllc"        : FLUIDS_RIEMANN_HLLC,
                  "exact"       : FLUIDS_RIEMANN_EXACT}

_fluidsystem_i     = inverse_dict(_fluidsystem)
_coordsystem_i     = inverse_dict(_coordsystem)
_equationofstate_i = inverse_dict(_equationofstate)
_riemannsolver_i   = inverse_dict(_riemannsolver)

class NegativeDensityCons(RuntimeError): pass
class NegativeDensityPrim(RuntimeError): pass
class NegativeEnergy(RuntimeError): pass
class NegativePressure(RuntimeError): pass
class SuperluminalVelocity(RuntimeError): pass
class ConsToPrimMaxIteration(RuntimeError): pass

_errors = {
  FLUIDS_ERROR_RIEMANN: RuntimeError,
  FLUIDS_ERROR_INCOMPLETE: RuntimeError,
  FLUIDS_ERROR_NEGATIVE_DENSITY_CONS: NegativeDensityCons,
  FLUIDS_ERROR_NEGATIVE_DENSITY_PRIM: NegativeDensityPrim,
  FLUIDS_ERROR_NEGATIVE_ENERGY: NegativeEnergy,
  FLUIDS_ERROR_NEGATIVE_PRESSURE: NegativePressure,
  FLUIDS_ERROR_SUPERLUMINAL: SuperluminalVelocity,
  FLUIDS_ERROR_C2P_MAXITER: ConsToPrimMaxIteration,
  FLUIDS_ERROR_NOT_IMPLEMENTED: NotImplementedError,
}

cdef class FluidDescriptor(object):
    """
    Class that describes the microphysics (equation of state) and required
    buffer sizes for a FluidState.
    """
    def __cinit__(self):
        self._c = fluids_descr_new()

    def __dealloc__(self):
        fluids_descr_del(self._c)

    def __init__(self, fluid='nrhyd', coordsystem='cartesian', eos='gammalaw',
                 gamma=1.4):
        fluids_descr_setfluid(self._c, _fluidsystem[fluid])
        fluids_descr_seteos(self._c, _equationofstate[eos])
        fluids_descr_setcoordsystem(self._c, _coordsystem[coordsystem])
        fluids_descr_setgamma(self._c, gamma)

    property fluid:
        def __get__(self):
            cdef int val
            fluids_descr_getfluid(self._c, &val)
            return _fluidsystem_i[val]

    property coordsystem:
        def __get__(self):
            cdef int val
            fluids_descr_getcoordsystem(self._c, &val)
            return _coordsystem_i[val]

    property eos:
        def __get__(self):
            cdef int val
            fluids_descr_geteos(self._c, &val)
            return _equationofstate_i[val]

    property gamma:
        def __get__(self):
            cdef double val
            fluids_descr_getgamma(self._c, &val)
            return val

    property rhobar:
        def __get__(self):
            cdef double val
            fluids_descr_getrhobar(self._c, &val)
            return val
        def __set__(self, double val):
            fluids_descr_setrhobar(self._c, val)

    property nprimitive:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_PRIMITIVE)
    property npassive:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_PASSIVE)
    property ngravity:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_GRAVITY)
    property nmagnetic:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_MAGNETIC)
    property nlocation:
        def __get__(self):
            return fluids_descr_getncomp(self._c, FLUIDS_LOCATION)

    def __reduce__(self):
        props = [getattr(self, k) for k in
                 ['fluid', 'coordsystem', 'eos', 'gamma']]
        state = {'rhobar': self.rhobar}
        return (FluidDescriptor, tuple(props), state)

    def __setstate__(self, props):
        for k, v in props.iteritems():
            setattr(self, k, v)


cdef class FluidState(object):
    """
    Class that holds fluid variables, and caches them for future calls. These
    objects will accumulate lots of memory unless the cache is disabled (which
    it is by default), or the erase_cache() method is used after each time a
    member function is invoked.
    """
    def __cinit__(self):
        self._c = fluids_state_new()

    def __dealloc__(self):
        fluids_state_del(self._c)

    def __init__(self, *args, **kwargs):
        cdef FluidDescriptor D
        try:
            D = args[0]
        except:
            D = FluidDescriptor(**kwargs)
        self._descr = D
        fluids_state_setdescr(self._c, D._c)
        self._np = D.nprimitive
        self._ns = D.npassive
        self._ng = D.ngravity
        self._nm = D.nmagnetic
        self._nl = D.nlocation
        if self._np: self.map_buffer('primitive', np.zeros(self._np))
        if self._ns: self.map_buffer('passive', np.zeros(self._ns))
        if self._ng: self.map_buffer('gravity', np.zeros(self._ng))
        if self._nm: self.map_buffer('magnetic', np.zeros(self._nm))
        if self._nl: self.map_buffer('location', np.zeros(self._nl))

    def map_buffer(self, key, buf):
        cdef long flag
        cdef int size
        cdef double *x
        if key == 'primitive':
            if buf.shape != (self._np,) or buf.dtype != np.double:
                raise ValueError("wrong size or type buffer")
            self._primitive = buf
            x = <double*>self._primitive.data
            size = self._np
            flag = FLUIDS_PRIMITIVE
            fluids_state_mapbuffer(self._c, x, flag)
        elif key == 'passive':
            if buf.shape != (self._ns,) or buf.dtype != np.double:
                raise ValueError("wrong size or type buffer")
            self._passive = buf
            x = <double*>self._passive.data
            size = self._ns
            flag = FLUIDS_PASSIVE
            fluids_state_mapbuffer(self._c, x, flag)
        elif key == 'gravity':
            if buf.shape != (self._ng,) or buf.dtype != np.double:
                raise ValueError("wrong size or type buffer")
            self._gravity = buf
            x = <double*>self._gravity.data
            size = self._ng
            flag = FLUIDS_GRAVITY
            fluids_state_mapbuffer(self._c, x, flag)
        elif key == 'magnetic':
            if buf.shape != (self._nm,) or buf.dtype != np.double:
                raise ValueError("wrong size or type buffer")
            self._magnetic = buf
            x = <double*>self._magnetic.data
            size = self._nm
            flag = FLUIDS_MAGNETIC
            fluids_state_mapbuffer(self._c, x, flag)
        elif key == 'location':
            if buf.shape != (self._nl,) or buf.dtype != np.double:
                raise ValueError("wrong size or type buffer")
            self._location = buf
            x = <double*>self._location.data
            size = self._nl
            flag = FLUIDS_LOCATION
            fluids_state_mapbuffer(self._c, x, flag)
        elif key == 'userflag':
            if buf.shape != (1,) or buf.dtype != np.int:
                raise ValueError("wrong size or type buffer")
            self._userflag = buf
            flag = FLUIDS_USERFLAG
            fluids_state_mapbufferuserflag(self._c, <int*>self._userflag.data)
        else:
            raise ValueError("bad buffer flag: " + key)


    property descriptor:
        def __get__(self):
            return self._descr
    property primitive:
        def __get__(self):
            return self._primitive
        def __set__(self, val):
            self._primitive[...] = val
    property passive:
        def __get__(self):
            return self._passive
        def __set__(self, val):
            self._passive[...] = val
    property gravity:
        def __get__(self):
            return self._gravity
        def __set__(self, val):
            self._gravity[...] = val
    property magnetic:
        def __get__(self):
            return self._magnetic
        def __set__(self, val):
            self._magnetic[...] = val
    property location:
        def __get__(self):
            return self._location
        def __set__(self, val):
            self._location[...] = val
    property userflag:
        def __get__(self):
            return self._userflag
        def __set__(self, val):
            self._userflag[...] = val

    def from_conserved(self, np.ndarray[np.double_t,ndim=1] x):
        if x.size != self._np: raise ValueError("wrong size input array")
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_fromcons(self._c, y, FLUIDS_CACHE_DEFAULT)
        if err != 0: raise _errors[err]("%s" % x)

    def conserved(self):
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, FLUIDS_CONSERVED)
        if err != 0: raise _errors[err]()
        return x

    def flux(self, dim=0):
        cdef int flag = [FLUIDS_FLUX0, FLUIDS_FLUX1, FLUIDS_FLUX2][dim]
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, flag)
        if err != 0: raise _errors[err]()
        return x

    def source_terms(self):
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, FLUIDS_SOURCETERMS)
        if err != 0: raise _errors[err]()
        return x

    def eigenvalues(self, dim=0):
        cdef int flag = [FLUIDS_EVAL0, FLUIDS_EVAL1, FLUIDS_EVAL2][dim]
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self._np)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, flag)
        if err != 0: raise _errors[err]()
        return x

    def left_eigenvectors(self, dim=0):
        cdef int flag = [FLUIDS_LEVECS0, FLUIDS_LEVECS1, FLUIDS_LEVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, flag)
        if err != 0: raise _errors[err]()
        return x

    def right_eigenvectors(self, dim=0):
        cdef int flag = [FLUIDS_REVECS0, FLUIDS_REVECS1, FLUIDS_REVECS2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, flag)
        if err != 0: raise _errors[err]()
        return x

    def jacobian(self, dim=0):
        cdef int flag = [FLUIDS_JACOBIAN0, FLUIDS_JACOBIAN1, FLUIDS_JACOBIAN2][dim]
        cdef np.ndarray[np.double_t,ndim=2] x = np.zeros([self._np]*2)
        cdef double *y = <double*>x.data
        cdef int err = fluids_state_derive(self._c, y, flag)
        if err != 0: raise _errors[err]()
        return x

    def sound_speed(self):
        cdef double cs2
        fluids_state_derive(self._c, &cs2, FLUIDS_SOUNDSPEEDSQUARED)
        return cs2**0.5


cdef class FluidStateVector(FluidState):
    def __init__(self, shape, *args, **kwargs):
        super(FluidStateVector, self).__init__(*args, **kwargs)
        shape = tuple(shape)
        self._states = np.ndarray(shape=shape, dtype=FluidState)

        self._primitive = np.zeros(shape + (self._np,))
        self._passive = np.zeros(shape + (self._ns,))
        self._gravity = np.zeros(shape + (self._ng,))
        self._magnetic = np.zeros(shape + (self._nm,))
        self._location = np.zeros(shape + (self._nl,))
        self._userflag = np.zeros(shape + (1,), dtype=np.int)

        cdef FluidState state
        cdef int n
        cdef np.ndarray P = self.primitive.reshape([self.states.size, self._np])
        cdef np.ndarray S = self.passive.reshape([self.states.size, self._ns])
        cdef np.ndarray G = self.gravity.reshape([self.states.size, self._ng])
        cdef np.ndarray M = self.magnetic.reshape([self.states.size, self._nm])
        cdef np.ndarray L = self.location.reshape([self.states.size, self._nl])
        cdef np.ndarray F = self.userflag.reshape([self.states.size, 1])

        for arr in [P, S, G, M, L]:
            assert not arr.flags['OWNDATA']

        for n in range(self.states.size):
            state = FluidState(self.descriptor)
            if self._np: state.map_buffer('primitive', P[n])
            if self._ns: state.map_buffer('passive', S[n])
            if self._ng: state.map_buffer('gravity', G[n])
            if self._nm: state.map_buffer('magnetic', M[n])
            if self._nl: state.map_buffer('location', L[n])
            state.map_buffer('userflag', F[n])
            self._states.flat[n] = state

    property shape:
        def __get__(self):
            return self.states.shape
    property size:
        def __get__(self):
            return self.states.size
    property flat:
        def __get__(self):
            return self.states.flat
    property states:
        def __get__(self):
            return self._states

    cdef _derive(self, long flag, int size):
        cdef tuple shape = self.states.shape + ((size,) if size > 1 else tuple())
        cdef np.ndarray ret = np.zeros(shape)
        cdef int n, e
        cdef FluidState S
        for n in range(self.states.size):
            S = self.states.flat[n]
            e = fluids_state_derive(S._c, <double*>ret.data + n*size, flag)
            if e != 0: self._userflag.flat[n] = e
        return ret

    def __getitem__(self, args):
        return self.states.__getitem__(args)

    def from_conserved(self, U):
        if U.shape != self.states.shape + (self._np,):
            raise ValueError("wrong size input array")
        cdef int n, e, numerr
        cdef FluidState S
        cdef np.ndarray[np.double_t] x = U.reshape([U.size])
        for n in range(self.states.size):
            S = self.states.flat[n]
            e = fluids_state_fromcons(S._c, <double*>x.data + n*self._np,
                                      FLUIDS_CACHE_DEFAULT)
            if e != 0: self._userflag.flat[n] = e
            numerr += (e != 0)
        return numerr

    def conserved(self):
        return self._derive(FLUIDS_CONSERVED, self._np)

    def source_terms(self):
        return self._derive(FLUIDS_SOURCETERMS, self._np)

    def flux(self, dim=0):
        cdef int flag = [FLUIDS_FLUX0, FLUIDS_FLUX1, FLUIDS_FLUX2][dim]
        return self._derive(flag, self._np)

    def eigenvalues(self, dim=0):
        cdef int flag = [FLUIDS_EVAL0, FLUIDS_EVAL1, FLUIDS_EVAL2][dim]
        return self._derive(flag, self._np)

    def left_eigenvectors(self, dim=0):
        raise NotImplementedError

    def right_eigenvectors(self, dim=0):
        raise NotImplementedError

    def sound_speed(self):
        cs2 = self._derive(FLUIDS_SOUNDSPEEDSQUARED, 1)
        return cs2**0.5


cdef class RiemannSolver(object):
    """
    Class which represents a two-state riemann solver.
    """
    def __cinit__(self):
        self._c = fluids_riemn_new()
        self.SL = None
        self.SR = None

    def __dealloc__(self):
        fluids_riemn_del(self._c)

    def __init__(self):
        fluids_riemn_setsolver(self._c, FLUIDS_RIEMANN_EXACT)

    property solver:
        def __get__(self):
            cdef int solver
            fluids_riemn_getsolver(self._c, &solver)
            return _riemannsolver_i[solver]
        def __set__(self, val):
            fluids_riemn_setsolver(self._c, _riemannsolver[val])

    def set_states(self, FluidState SL, FluidState SR):
        if SL._descr is not SR._descr:
            raise ValueError("different fluid descriptor on left and right")
        self.SL = SL # hold onto these so they're not deleted
        self.SR = SR
        fluids_riemn_setdim(self._c, 0)
        fluids_riemn_setstateL(self._c, SL._c)
        fluids_riemn_setstateR(self._c, SR._c)
        fluids_riemn_execute(self._c)

    def sample(self, double s):
        if self.SL is None or self.SR is None:
            raise ValueError("solver needs a need a left and right state")
        cdef FluidState S = FluidState(self.SL._descr)
        fluids_riemn_sample(self._c, S._c, s)
        return S
