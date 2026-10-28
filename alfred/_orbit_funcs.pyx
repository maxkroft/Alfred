import numpy as np
cimport numpy as np

cdef extern from "orbit_funcs.c":

    double solve_Kepler(const double t, const double p, const double tp, const double e)

    void rvModel(const double par[], const double t[], const int n, double rv[])
    
    void astrometryModel(const double par[], const double t[], const double psi[], const int n, double x[], double y[], double eta[])


def _solve_Kepler(double t, double p, double tp, double e):

    return solve_Kepler(t, p, tp, e)


def _rvModel(par, t):

    if not par.flags['C_CONTIGUOUS']:
        par = np.ascontiguousarray(par)

    cdef double[::1] par_memview = par

    if not t.flags['C_CONTIGUOUS']:
        t = np.ascontiguousarray(t)

    cdef double[::1] t_memview = t

    cdef int n = t_memview.shape[0]

    cdef np.ndarray rv = np.ascontiguousarray(np.zeros(n))

    cdef double[::1] rv_memview = rv

    rvModel(&par_memview[0], &t_memview[0], n, &rv_memview[0])

    return rv


def _astrometryModel(par, t, psi):

    if not par.flags['C_CONTIGUOUS']:
        par = np.ascontiguousarray(par)

    cdef double[::1] par_memview = par

    if not t.flags['C_CONTIGUOUS']:
        t = np.ascontiguousarray(t)

    cdef double[::1] t_memview = t

    if not psi.flags['C_CONTIGUOUS']:
        psi = np.ascontiguousarray(psi)

    cdef double[::1] psi_memview = psi

    cdef int n = t_memview.shape[0]

    cdef np.ndarray x = np.ascontiguousarray(np.zeros(n))

    cdef double[::1] x_memview = x

    cdef np.ndarray y = np.ascontiguousarray(np.zeros(n))

    cdef double[::1] y_memview = y

    cdef np.ndarray eta = np.ascontiguousarray(np.zeros(n))

    cdef double[::1] eta_memview = eta

    astrometryModel(&par_memview[0], &t_memview[0], &psi_memview[0], n, &x_memview[0], &y_memview[0], &eta_memview[0])

    return x, y, eta