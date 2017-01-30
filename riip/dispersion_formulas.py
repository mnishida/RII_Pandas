#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logging import getLogger
import typing
from collections import Sequence
from itertools import islice
import numpy as np
from math import nan
import pandas as pd
from scipy.interpolate import interp1d

logger = getLogger(__package__)
PandasDataFrame = typing.NewType('PandasDataFrame', pd.DataFrame)
FloatNdarray = typing.NewType(
    'FloatNdarray', typing.Union[float, np.ndarray])


class DispersionFormula:
    """This class provide dispersion formula defined
     in refractiveindex.info database.

     Attributes:
         exp_data (PandaDataFrame): The experimental data set.
         formulas (dict[int, Callable]): A dict of functions for the formulas.
     """

    def __init__(self, catalog: PandasDataFrame, exp_data: PandasDataFrame):
        """Initialize DispersionFormula

        Args:
            catalog: The catalog data set.
            exp_data: The experimental data set.
        """
        self.catalog = catalog
        self.exp_data = exp_data
        self.formulas = {i: func for i, func in enumerate(
            [formula_1, formula_2, formula_3, formula_4, formula_5,
             formula_6, formula_7, formula_8, formula_9], 1)}

    def __call__(self, x: FloatNdarray) \
            -> typing.Tuple[FloatNdarray, FloatNdarray]:
        """Return n and k for given wavelength

        Args:
            x: Wavelength.
        """
        wl_min = self.catalog['wl_min']
        wl_max = self.catalog['wl_max']
        if isinstance(x, (Sequence, np.ndarray)):
            x_min = min(x)
            x_max = max(x)
        else:
            x_min = x_max = x
        if x_min < wl_min or x_max > wl_max:
            raise ValueError(
                'Wavelength is out of bounds [{} {}][um]'.format(
                    wl_min, wl_max))

        formula = self.catalog['formula']
        tabulated = self.catalog['tabulated']
        if tabulated != tabulated:
            n = k = nan
        else:
            if formula == formula:
                n = self.formulas[int(formula)](x, self.exp_data['c'].values)
            elif 'n' in tabulated:
                wls_n = self.exp_data['wl_n'].values
                ns = self.exp_data['n'].values
                if len(ns) == 1:
                    n = ns * np.ones_like(x)
                else:
                    n = interp1d(
                        wls_n, ns, kind='linear', bounds_error=False,
                        fill_value=(ns[0], ns[-1]), assume_sorted=True)(x)
            else:
                n = nan
            if 'k' in tabulated:
                wls_k = self.exp_data['wl_k'].values
                ks = self.exp_data['k'].values
                if len(ks) == 1:
                    k = ks.values * np.ones_like(x)
                else:
                    k = interp1d(
                        wls_k, ks, kind='linear', bounds_error=False,
                        fill_value=(ks[0], ks[-1]), assume_sorted=True)(x)
            else:
                k = nan
        return n, k


def formula_1(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    x_sqr = x ** 2
    n_sqr = 1 + cs[0]
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        n_sqr += c1 * x_sqr / (x_sqr - c2 ** 2)
    return np.sqrt(n_sqr)


def formula_2(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    x_sqr = x ** 2
    n_sqr = 1 + cs[0]
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        n_sqr += c1 * x_sqr / (x_sqr - c2)
    return np.sqrt(n_sqr)


def formula_3(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    n_sqr = cs[0]
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        n_sqr += c1 * x ** c2
    return np.sqrt(n_sqr)


def formula_4(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    n_sqr = (cs[0] + cs[1] * x ** cs[2] / (x ** 2 - cs[3] ** cs[4]) +
             cs[5] * x ** cs[6] / (x ** 2 - cs[7] ** cs[8]))
    for c1, c2 in zip(islice(cs, 9, None, 2), islice(cs, 10, None, 2)):
        n_sqr += c1 * x ** c2
    return np.sqrt(n_sqr)


def formula_5(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    n = cs[0]
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        n += c1 * x ** c2
    return n


def formula_6(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    x_m2 = 1 / x ** 2
    n = 1 + cs[0]
    for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
        n += c1 / (c2 - x_m2)
    return n


def formula_7(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    x_sqr = x ** 2
    n = (cs[0] + cs[1] / (x_sqr - 0.028) + cs[2] / (x_sqr - 0.028) ** 2 +
         cs[3] * x_sqr + cs[4] * x_sqr ** 2 + cs[5] * x_sqr ** 3)
    return n


def formula_8(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    x_sqr = x ** 2
    a = cs[0] + cs[1] * x_sqr / (x_sqr - cs[2]) + cs[3] * x_sqr
    n_sqr = (1 + 2 * a) / (1 - a)
    return np.sqrt(n_sqr)


def formula_9(x: FloatNdarray, cs: FloatNdarray) -> FloatNdarray:
    n_sqr = (cs[0] + cs[1] / (x ** 2 - cs[2]) +
             cs[3] * (x - cs[4]) / ((x - cs[4]) ** 2 + cs[5]))
    return np.sqrt(n_sqr)
