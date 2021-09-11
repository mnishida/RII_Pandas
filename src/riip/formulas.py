from __future__ import annotations

from collections.abc import Callable
from itertools import islice

import numpy as np
from scipy import constants
from scipy.special import wofz

from .formulas_cython import (
    formula_1_cython,
    formula_2_cython,
    formula_3_cython,
    formula_4_cython,
    formula_5_cython,
    formula_6_cython,
    formula_7_cython,
    formula_8_cython,
    formula_9_cython,
    formula_21_cython,
    formula_22_cython,
)

UNIT = constants.h * constants.c * 1e6 / constants.e


def _formula_1(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    x_sqr: np.ndarray = x ** 2
    n_sqr = 1 + cs[0]
    c1 = 0.0
    c2 = 0.0
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        if c1 == 0.0:
            break
        n_sqr += c1 * x_sqr / (x_sqr - c2 ** 2)
    return np.sqrt(n_sqr * (n_sqr > 0))


def _formula_2(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    x_sqr = x ** 2
    n_sqr: float = 1 + cs[0]
    c1: float = 0.0
    c2: float = 0.0
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        if c1 == 0.0:
            break
        n_sqr += c1 * x_sqr / (x_sqr - c2)
    return np.sqrt(n_sqr * (n_sqr > 0))


def _formula_3(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    n_sqr: float = cs[0]
    c1 = 0.0
    c2 = 0.0
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        if c1 == 0.0:
            break
        n_sqr += c1 * x ** c2
    return np.sqrt(n_sqr * (n_sqr > 0))


def _formula_4(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    n_sqr = (
        cs[0]
        + cs[1] * x ** cs[2] / (x ** 2 - cs[3] ** cs[4])
        + cs[5] * x ** cs[6] / (x ** 2 - cs[7] ** cs[8])
    )
    c1 = 0.0
    c2 = 0.0
    for c1, c2 in zip(islice(cs, 9, None, 2), islice(cs, 10, None, 2)):
        if c1 == 0.0:
            break
        n_sqr += c1 * x ** c2
    return np.sqrt(n_sqr * (n_sqr > 0))


def _formula_5(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    n = cs[0]
    c1 = 0.0
    c2 = 0.0
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        if c1 == 0.0:
            break
        n += c1 * x ** c2
    return n


def _formula_6(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    x_m2 = 1 / x ** 2
    n = 1 + cs[0]
    c1 = 0.0
    c2 = 0.0
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        if c1 == 0.0:
            break
        n += c1 / (c2 - x_m2)
    return n


def _formula_7(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    x_sqr = x ** 2
    n = (
        cs[0]
        + cs[1] / (x_sqr - 0.028)
        + cs[2] / (x_sqr - 0.028) ** 2
        + cs[3] * x_sqr
        + cs[4] * x_sqr ** 2
        + cs[5] * x_sqr ** 3
    )
    return n


def _formula_8(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    x_sqr = x ** 2
    a = cs[0] + cs[1] * x_sqr / (x_sqr - cs[2]) + cs[3] * x_sqr
    n_sqr = (1 + 2 * a) / (1 - a)
    return np.sqrt(n_sqr * (n_sqr > 0))


def _formula_9(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    n_sqr = (
        cs[0]
        + cs[1] / (x ** 2 - cs[2])
        + cs[3] * (x - cs[4]) / ((x - cs[4]) ** 2 + cs[5])
    )
    return np.sqrt(n_sqr * (n_sqr > 0))


def _formula_21(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    w = UNIT / x
    eb = cs[0]
    f0 = cs[1]
    g0 = cs[2]
    wp = cs[3]
    eps = eb - f0 * wp ** 2 / (w ** 2 + 1j * w * g0)
    for fj, gj, wj in zip(
        islice(cs, 4, None, 3), islice(cs, 5, None, 3), islice(cs, 6, None, 3)
    ):
        if fj == 0:
            break
        eps -= fj * wp ** 2 / (w ** 2 - wj ** 2 + 1j * w * gj)
    return eps


def _formula_22(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
    w = UNIT / x
    eb = cs[0]
    f0 = cs[1]
    g0 = cs[2]
    wp = cs[3]
    c = 1j * np.sqrt(np.pi) / (2 * np.sqrt(2))
    eps = eb - f0 * wp ** 2 / (w ** 2 + 1j * w * g0)
    for fj, gj, wj, sj in zip(
        islice(cs, 4, None, 4),
        islice(cs, 5, None, 4),
        islice(cs, 6, None, 4),
        islice(cs, 7, None, 4),
    ):
        if fj == 0:
            break
        aj = np.sqrt(w * (w + 1j * gj))
        sj_inv = 1 / sj if sj != 0 else 0
        eps += (
            c
            * fj
            * wp ** 2
            / aj
            * sj_inv
            * (
                wofz((aj - wj) / np.sqrt(2) * sj_inv)
                + wofz((aj + wj) / np.sqrt(2) * sj_inv)
            )
        )
    return eps


formulas_numpy_dict: dict[int, Callable] = {
    1: _formula_1,
    2: _formula_2,
    3: _formula_3,
    4: _formula_4,
    5: _formula_5,
    6: _formula_6,
    7: _formula_7,
    8: _formula_8,
    9: _formula_9,
    21: _formula_21,
    22: _formula_22,
}

formulas_cython_dict: dict[int, Callable] = {
    1: formula_1_cython,
    2: formula_2_cython,
    3: formula_3_cython,
    4: formula_4_cython,
    5: formula_5_cython,
    6: formula_6_cython,
    7: formula_7_cython,
    8: formula_8_cython,
    9: formula_9_cython,
    21: formula_21_cython,
    22: formula_22_cython,
}
