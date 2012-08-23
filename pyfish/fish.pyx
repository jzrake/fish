
cimport fish
cimport numpy as np
import numpy as np

cdef class FishState(object):
    def __cinit__(self):
        self._c = fish_new()

    def __dealloc__(self):
        fish_del(self._c)

    def __init__(self):
        pass
