
from pyfluids.fluids cimport *
cimport fish
cimport numpy as np
import numpy as np

cdef extern from "fish.h":
    cdef int FISH_NONE           =   -42
    cdef int FISH_PLM            =   -43
    cdef int FISH_WENO5          =   -44

    struct fish_state

    fish_state *fish_new()
    int fish_del(fish_state *S)

    int fish_evolve(fish_state *S, double dt)
    int fish_intercellflux(fish_state *S, fluids_state **fluid, double *F, int N,
                           int dim)
    int fish_setfluid(fish_state *S, int fluid)
    int fish_setriemannsolver(fish_state *S, int riemannsolver)
    int fish_setreconstruction(fish_state *S, int reconstruction)
    int fish_setplmtheta(fish_state *S, double plmtheta)
    int fish_getfluid(fish_state *S, int *fluid)
    int fish_getriemannsolver(fish_state *S, int *riemannsolver)
    int fish_getreconstruction(fish_state *S, int *reconstruction)
    int fish_getplmtheta(fish_state *S, double *plmtheta)


cdef class FishSolver(object):
    cdef fish_state *_c
