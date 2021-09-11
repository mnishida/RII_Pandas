import numpy as np
from scipy import constants

cimport cython
cimport numpy as np
from libc.math cimport M_PI, sqrt
from scipy.special.cython_special cimport wofz

ctypedef np.complex128_t cdouble

cdef extern from "<complex>" namespace "std" nogil:
    cdouble csqrt "sqrt" (cdouble z)
    double creal "real" (cdouble z)

cdef:
    double UNIT = constants.h * constants.c * 1e6 / constants.e / (2 * M_PI)
    cdouble C_I = 1.0j
    cdouble C = C_I * sqrt(M_PI) / 2
    double RT2_INV = 1 / sqrt(2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_1_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        double x_sqr = (2 * M_PI / creal(w)) ** 2
        double n_sqr = 1 + cs[0]
        double c1, c2

    for i in range(11):
        c1 = cs[2 * i + 1]
        c2 = cs[2 * i + 2]
        if c1 == 0:
            break
        n_sqr += c1 * x_sqr / (x_sqr - c2 ** 2)
    return sqrt(n_sqr * (n_sqr > 0))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_2_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        double x_sqr = (2 * M_PI / creal(w)) ** 2
        double n_sqr = 1 + cs[0]
        double c1, c2
    for i in range(11):
        c1 = cs[2 * i + 1]
        c2 = cs[2 * i + 2]
        if c1 == 0:
            break
        n_sqr += c1 * x_sqr / (x_sqr - c2)
    return sqrt(n_sqr * (n_sqr > 0))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_3_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        double x = 2 * M_PI / creal(w)
        double n_sqr = cs[0]
        double c1, c2
    for i in range(11):
        c1 = cs[2 * i + 1]
        c2 = cs[2 * i + 2]
        if c1 == 0:
            break
        n_sqr += c1 * x ** c2
    return sqrt(n_sqr * (n_sqr > 0))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_4_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        double x = 2 * M_PI / creal(w)
        double n_sqr = (
            cs[0]
            + cs[1] * x ** cs[2] / (x ** 2 - cs[3] ** cs[4])
            + cs[5] * x ** cs[6] / (x ** 2 - cs[7] ** cs[8])
        )
    for i in range(7):
        c1 = cs[2 * i + 9]
        c2 = cs[2 * i + 10]
        if c1 == 0:
            break
        n_sqr += c1 * x ** c2
    return sqrt(n_sqr * (n_sqr > 0))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_5_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        double n = cs[0]
        double x = 2 * M_PI / creal(w)
    for i in range(11):
        c1 = cs[2 * i + 1]
        c2 = cs[2 * i + 2]
        if c1 == 0:
            break
        n += c1 * x ** c2
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_6_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        double x_m2 = (creal(w) / (2 * M_PI)) ** 2
        double n = 1 + cs[0]
    for i in range(11):
        c1 = cs[2 * i + 1]
        c2 = cs[2 * i + 2]
        if c1 == 0:
            break
        n += c1 / (c2 - x_m2)
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_7_cython(cdouble w, double[::1] cs):
    cdef:
        double x_sqr = (2 * M_PI / creal(w)) ** 2
        double n = (
            cs[0]
            + cs[1] / (x_sqr - 0.028)
            + cs[2] / (x_sqr - 0.028) ** 2
            + cs[3] * x_sqr
            + cs[4] * x_sqr ** 2
            + cs[5] * x_sqr ** 3
        )
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_8_cython(cdouble w, double[::1] cs):
    cdef:
        double x_sqr = (2 * M_PI / creal(w)) ** 2
        double a = cs[0] + cs[1] * x_sqr / (x_sqr - cs[2]) + cs[3] * x_sqr
        double n_sqr = (1 + 2 * a) / (1 - a)
    return sqrt(n_sqr * (n_sqr > 0))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_9_cython(cdouble w, double[::1] cs):
    cdef:
        double x = 2 * M_PI / creal(w)
        double n_sqr = (
                cs[0]
                + cs[1] / (x ** 2 - cs[2])
                + cs[3] * (x - cs[4]) / ((x - cs[4]) ** 2 + cs[5])
            )
    return sqrt(n_sqr * (n_sqr > 0))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_21_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        cdouble _w = w * UNIT
        double eb = cs[0]
        double f0 = cs[1]
        double g0 = cs[2]
        double wp = cs[3]
        cdouble eps = eb - f0 * wp ** 2 / (_w ** 2 + C_I * _w * g0)
        double fj, gj, wj,

    for i in range(1, 7):
        fj = cs[3 * i + 1]
        gj = cs[3 * i + 2]
        wj = cs[3 * i + 3]
        if fj ==0:
            break
        eps -= fj * wp ** 2 / (_w ** 2 - wj ** 2 + C_I * _w * gj)
    return eps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def formula_22_cython(cdouble w, double[::1] cs):
    cdef:
        int i
        cdouble _w = w * UNIT
        double eb = cs[0]
        double f0 = cs[1]
        double g0 = cs[2]
        double wp = cs[3]
        cdouble eps = eb - f0 * wp ** 2 / (_w ** 2 + 1j * _w * g0)
        cdouble aj
        double fj, gj, wj, sj, sj_inv

    for i in range(1, 6):
        fj = cs[4 * i]
        gj = cs[4 * i + 1]
        wj = cs[4 * i + 2]
        sj = cs[4 * i + 3]
        aj = csqrt(_w * (_w + 1j * gj))
        sj_inv = RT2_INV / sj if sj != 0 else 0
        if fj ==0:
            break
        eps += (
            C * fj * wp ** 2 / aj * sj_inv
            * (
                wofz((aj - wj) * sj_inv)
                + wofz((aj + wj) * sj_inv)
            )
        )
    return eps
