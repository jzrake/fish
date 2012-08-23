
cdef extern from "fluids.h":
    struct fluid_state

cdef extern from "fish.h":
    cdef int FISH_NONE           =   -42
    cdef int FISH_PLM            =   -43
    cdef int FISH_WENO5          =   -44

    struct fish_state

    fish_state *fish_new()
    int fish_del(fish_state *S)

    int fish_evolve(fish_state *S, double dt)
    int fish_intercellflux(fish_state *S, fluid_state **fluid, double *F, int N,
                           int dim)
    int fish_setfluid(fish_state *S, int fluid)
    int fish_setriemannsolver(fish_state *S, int riemannsolver)
    int fish_setreconstruction(fish_state *S, int reconstruction)
    int fish_setplmtheta(fish_state *S, double plmtheta)

cdef class FishState(object):
    cdef fish_state *_c

