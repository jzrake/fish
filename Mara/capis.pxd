
cimport numpy as np

cdef extern from "fluids.h":
    enum:
        FLUIDS_PRIMITIVE         =  1<<1,
        FLUIDS_PASSIVE           =  1<<2,
        FLUIDS_GRAVITY           =  1<<3,
        FLUIDS_MAGNETIC          =  1<<4,
        FLUIDS_LOCATION          =  1<<5,
        FLUIDS_USERFLAG          =  1<<6,
        FLUIDS_CONSERVED         =  1<<7,
        FLUIDS_SOURCETERMS       =  1<<8,
        FLUIDS_FOURVELOCITY      =  1<<9,
        FLUIDS_FLUX0             =  1<<10,
        FLUIDS_FLUX1             =  1<<11,
        FLUIDS_FLUX2             =  1<<12,
        FLUIDS_EVAL0             =  1<<13,
        FLUIDS_EVAL1             =  1<<14,
        FLUIDS_EVAL2             =  1<<15,
        FLUIDS_LEVECS0           =  1<<16,
        FLUIDS_LEVECS1           =  1<<17,
        FLUIDS_LEVECS2           =  1<<18,
        FLUIDS_REVECS0           =  1<<19,
        FLUIDS_REVECS1           =  1<<20,
        FLUIDS_REVECS2           =  1<<21,
        FLUIDS_JACOBIAN0         =  1<<22,
        FLUIDS_JACOBIAN1         =  1<<23,
        FLUIDS_JACOBIAN2         =  1<<24,
        FLUIDS_SOUNDSPEEDSQUARED =  1<<25,
        FLUIDS_TEMPERATURE       =  1<<26,
        FLUIDS_SPECIFICENTHALPY  =  1<<27,
        FLUIDS_SPECIFICINTERNAL  =  1<<28,
        FLUIDS_FLAGSALL          = (1<<30) - 1,
        FLUIDS_FLUXALL     = FLUIDS_FLUX0|FLUIDS_FLUX1|FLUIDS_FLUX2,
        FLUIDS_EVALSALL    = FLUIDS_EVAL0|FLUIDS_EVAL1|FLUIDS_EVAL2,
        FLUIDS_LEVECSALL   = FLUIDS_LEVECS0|FLUIDS_LEVECS1|FLUIDS_LEVECS2,
        FLUIDS_REVECSALL   = FLUIDS_REVECS0|FLUIDS_REVECS1|FLUIDS_REVECS2,
        FLUIDS_JACOBIANALL = FLUIDS_JACOBIAN0|FLUIDS_JACOBIAN1|FLUIDS_JACOBIAN2,

    enum:
        FLUIDS_SUCCESS,
        FLUIDS_ERROR_BADARG,
        FLUIDS_ERROR_BADREQUEST,
        FLUIDS_ERROR_RIEMANN,
        FLUIDS_ERROR_INCOMPLETE,
        FLUIDS_ERROR_NEGATIVE_DENSITY_CONS,
        FLUIDS_ERROR_NEGATIVE_DENSITY_PRIM,
        FLUIDS_ERROR_NEGATIVE_ENERGY,
        FLUIDS_ERROR_NEGATIVE_PRESSURE,
        FLUIDS_ERROR_SUPERLUMINAL,
        FLUIDS_ERROR_C2P_MAXITER,
        FLUIDS_ERROR_NOT_IMPLEMENTED,

        FLUIDS_COORD_CARTESIAN,
        FLUIDS_COORD_SPHERICAL,
        FLUIDS_COORD_CYLINDRICAL,

        FLUIDS_SCADV, # Scalar advection
        FLUIDS_SCBRG, # Burgers equation
        FLUIDS_SHWAT, # Shallow water equations
        FLUIDS_NRHYD, # Euler equations
        FLUIDS_GRAVS, # Gravitating Euler equations (with source terms on p and E)
        FLUIDS_GRAVP, # " "                         (no source term on p)
        FLUIDS_GRAVE, # " "                         (no source terms at all)
        FLUIDS_SRHYD, # Special relativistic
        FLUIDS_URHYD, # Ultra relativistic
        FLUIDS_GRHYD, # General relativistic
        FLUIDS_NRMHD, # Magnetohydrodynamic (MHD)
        FLUIDS_SRMHD, # Special relativistic MHD
        FLUIDS_GRMHD, # General relativistic MHD

        FLUIDS_EOS_GAMMALAW,
        FLUIDS_EOS_TABULATED,

        FLUIDS_RIEMANN_HLL,
        FLUIDS_RIEMANN_HLLC,
        FLUIDS_RIEMANN_EXACT,

        FLUIDS_CACHE_DEFAULT,
        FLUIDS_CACHE_NOTOUCH,
        FLUIDS_CACHE_CREATE,
        FLUIDS_CACHE_STEAL,
        FLUIDS_CACHE_RESET,
        FLUIDS_CACHE_ERASE


    struct fluids_descr
    struct fluids_cache
    struct fluids_state
    struct fluids_riemn


    # fluids_descr member functions
    fluids_descr *fluids_descr_new()
    int fluids_descr_del(fluids_descr *D)
    int fluids_descr_getfluid(fluids_descr *D, int *fluid)
    int fluids_descr_setfluid(fluids_descr *D, int fluid)
    int fluids_descr_geteos(fluids_descr *D, int *eos)
    int fluids_descr_seteos(fluids_descr *D, int eos)
    int fluids_descr_getcoordsystem(fluids_descr *D, int *coordsystem)
    int fluids_descr_setcoordsystem(fluids_descr *D, int coordsystem)
    int fluids_descr_getgamma(fluids_descr *D, double *gam)
    int fluids_descr_setgamma(fluids_descr *D, double gam)
    int fluids_descr_getrhobar(fluids_descr *D, double *rhobar)
    int fluids_descr_setrhobar(fluids_descr *D, double rhobar)
    int fluids_descr_getncomp(fluids_descr *D, long flag)


    # fluids_state member functions
    fluids_state *fluids_state_new()
    int fluids_state_del(fluids_state *S)
    int fluids_state_setdescr(fluids_state *S, fluids_descr *D)
    int fluids_state_getattr(fluids_state *S, double *x, long flag)
    int fluids_state_setattr(fluids_state *S, double *x, long flag)
    int fluids_state_fromcons(fluids_state *S, double *U, int cache)
    int fluids_state_derive(fluids_state *S, double *x, long flag)
    int fluids_state_getcached(fluids_state *S, double *x, long flag)
    int fluids_state_mapbuffer(fluids_state *S, double *buffer, long flag)
    int fluids_state_mapbufferuserflag(fluids_state *S, int *buffer)
    int fluids_state_getuserflag(fluids_state *S, int *x)
    int fluids_state_setuserflag(fluids_state *S, int *x)
    int fluids_state_cache(fluids_state *S, int operation)


    # fluids_riemn member functions
    fluids_riemn *fluids_riemn_new()
    int fluids_riemn_del(fluids_riemn *R)
    int fluids_riemn_setstateL(fluids_riemn *R, fluids_state *S)
    int fluids_riemn_setstateR(fluids_riemn *R, fluids_state *S)
    int fluids_riemn_setdim(fluids_riemn *R, int dim)
    int fluids_riemn_execute(fluids_riemn *R)
    int fluids_riemn_sample(fluids_riemn *R, fluids_state *S, double s)
    int fluids_riemn_setsolver(fluids_riemn *R, int solver)
    int fluids_riemn_getsolver(fluids_riemn *R, int *solver)


cdef extern from "fish.h":
    enum:
        FISH_PCM, # piecewise constant reconstruction
        FISH_PLM, # piecewise linear reconstruction
        FISH_WENO5, # weno-5 reconstruction
        FISH_GODUNOV, # conservative finite volume Riemann-solver intercell fluxes
        FISH_SPECTRAL, # conservative finite differencing of characteristic fields
        FISH_DIFFUSION, # apply Lax-Friedrichs diffusion (for when things get dicey)

        # ---------------------------------------------------------------------------
        # smoothness indicators for WENO reconstruction
        # ---------------------------------------------------------------------------
        FISH_ISK_JIANGSHU96, # original smoothness indicator of Jiang & Shu (1996)
        FISH_ISK_BORGES08, # improved by Borges (2008) NOTE: might be 4th order
        FISH_ISK_SHENZHA10, # improved by Shen & Zha (2010)

        # ---------------------------------------------------------------------------
        # names of parameters for solver description
        # ---------------------------------------------------------------------------

        # ------------------
        # integer parameters
        # ------------------
        FISH_SOLVER_TYPE,
        FISH_RIEMANN_SOLVER,
        FISH_RECONSTRUCTION,
        FISH_SMOOTHNESS_INDICATOR,

        # -----------------
        # double parameters
        # -----------------
        FISH_PLM_THETA, # [1 -> 2 (most aggressive)]
        FISH_SHENZHA10_PARAM, # [0 -> ~100 (most aggressive)]

        FISH_ERROR_BADARG,

    struct fish_state

    fish_state *fish_new()
    int fish_del(fish_state *S)
    int fish_evolve(fish_state *S, double dt)
    int fish_intercellflux(fish_state *S, fluids_state **fluid, double *F, int N,
                           int dim)
    int fish_timederivative(fish_state *S, fluids_state **fluid,
                            int ndim, int *shape, double *dx, double *L)
    int fish_getparami(fish_state *S, int *param, long flag)
    int fish_setparami(fish_state *S, int param, long flag)
    int fish_getparamd(fish_state *S, double *param, long flag)
    int fish_setparamd(fish_state *S, double param, long flag)



cdef class FluidDescriptor(object):
    cdef fluids_descr *_c


cdef class FluidState(object):
    cdef fluids_state *_c
    cdef FluidDescriptor _descr
    cdef int _np
    cdef int _ns
    cdef int _ng
    cdef int _nm
    cdef int _nl
    cdef np.ndarray _states
    cdef np.ndarray _primitive
    cdef np.ndarray _passive
    cdef np.ndarray _gravity
    cdef np.ndarray _magnetic
    cdef np.ndarray _location
    cdef np.ndarray _userflag


cdef class FluidStateVector(FluidState):
    cdef _derive(self, long flag, int sz)


cdef class RiemannSolver(object):
    cdef fluids_riemn *_c
    cdef FluidState SL
    cdef FluidState SR


cdef class FishSolver(object):
    cdef fish_state *_c
