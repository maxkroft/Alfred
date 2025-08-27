import numpy as np
cimport numpy as np

cdef extern from "rv_func.c":

    void rvModel(const double par[], const double t[], const int n, double rv[])


def _rvModel(par, t):

    if not par.flags['C_CONTIGUOUS']:
        par = np.ascontiguousarray(par)

    cdef double[::1] par_memview = par

    if not t.flags['C_CONTIGUOUS']:
        t = np.ascontiguousarray(t)

    cdef double[::1] t_memview = t

    cdef np.ndarray rv = np.ascontiguousarray(np.zeros(t_memview.shape[0]))

    cdef double[::1] rv_memview = rv

    rvModel(&par_memview[0], &t_memview[0], t_memview.shape[0], &rv_memview[0])

    return rv