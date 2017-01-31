#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logging import getLogger
import typing
from collections import Sequence
from itertools import islice
import numpy as np
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
        self.formulas = {'1': self.formula_1, '2': self.formula_2,
                         '3': self.formula_3, '4': self.formula_4,
                         '5': self.formula_5, '6': self.formula_6,
                         '7': self.formula_7, '8': self.formula_8,
                         '9': self.formula_9}

    def func_n(self, x: FloatNdarray) -> FloatNdarray:
        """Return n for given wavelength

        Args:
            x: Wavelength.
        """
        wl_n_min = self.catalog['wl_n_min']
        wl_n_max = self.catalog['wl_n_max']
        if isinstance(x, (Sequence, np.ndarray)):
            x_min = min(x)
            x_max = max(x)
        else:
            x_min = x_max = x
        if x_min < wl_n_min or x_max > wl_n_max:
            raise ValueError(
                'Wavelength is out of bounds [{} {}][um]'.format(
                    wl_n_min, wl_n_max))

        formula = self.catalog['formula']
        tabulated = self.catalog['tabulated']
        if formula != '':
            return self.formulas[formula](x)
        elif tabulated == tabulated:
            if 'n' in tabulated:
                num_n = self.catalog['num_n']
                wls_n = self.exp_data['wl_n'].values[:num_n]
                ns = self.exp_data['n'].values[:num_n]
                if len(ns) == 1:
                    return ns * np.ones_like(x)
                else:
                    return interp1d(
                        wls_n, ns, kind='linear', bounds_error=False,
                        fill_value=(ns[0], ns[-1]), assume_sorted=True)(x)
        else:
            return np.empty_like(x)

    def func_k(self, x: FloatNdarray) -> FloatNdarray:
        """Return n for given wavelength

        Args:
            x: Wavelength.
        """
        wl_k_min = self.catalog['wl_k_min']
        wl_k_max = self.catalog['wl_k_max']
        if isinstance(x, (Sequence, np.ndarray)):
            x_min = min(x)
            x_max = max(x)
        else:
            x_min = x_max = x
        if x_min < wl_k_min or x_max > wl_k_max:
            raise ValueError(
                'Wavelength is out of bounds [{} {}][um]'.format(
                    wl_k_min, wl_k_max))
        tabulated = self.catalog['tabulated']
        if tabulated == tabulated:
            if 'k' in tabulated:
                num_k = self.catalog['num_k']
                wls_k = self.exp_data['wl_k'].values[:num_k]
                ks = self.exp_data['k'].values[:num_k]
                if len(ks) == 1:
                    return ks * np.ones_like(x)
                else:
                    return interp1d(
                        wls_k, ks, kind='linear', bounds_error=False,
                        fill_value=(ks[0], ks[-1]), assume_sorted=True)(x)
        else:
            return np.empty_like(x)

    def formula_1(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        x_sqr = x ** 2
        n_sqr = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x_sqr / (x_sqr - c2 ** 2)
        return np.sqrt(n_sqr)

    def formula_2(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        x_sqr = x ** 2
        n_sqr = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x_sqr / (x_sqr - c2)
        return np.sqrt(n_sqr)

    def formula_3(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        n_sqr = cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x ** c2
        return np.sqrt(n_sqr)

    def formula_4(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        n_sqr = (cs[0] + cs[1] * x ** cs[2] / (x ** 2 - cs[3] ** cs[4]) +
                 cs[5] * x ** cs[6] / (x ** 2 - cs[7] ** cs[8]))
        for c1, c2 in zip(islice(cs, 9, None, 2), islice(cs, 10, None, 2)):
            n_sqr += c1 * x ** c2
        return np.sqrt(n_sqr)

    def formula_5(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        n = cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n += c1 * x ** c2
        return n

    def formula_6(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        x_m2 = 1 / x ** 2
        n = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n += c1 / (c2 - x_m2)
        return n

    def formula_7(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        x_sqr = x ** 2
        n = (cs[0] + cs[1] / (x_sqr - 0.028) + cs[2] / (x_sqr - 0.028) ** 2 +
             cs[3] * x_sqr + cs[4] * x_sqr ** 2 + cs[5] * x_sqr ** 3)
        return n

    def formula_8(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        x_sqr = x ** 2
        a = cs[0] + cs[1] * x_sqr / (x_sqr - cs[2]) + cs[3] * x_sqr
        n_sqr = (1 + 2 * a) / (1 - a)
        return np.sqrt(n_sqr)

    def formula_9(self, x: FloatNdarray) -> FloatNdarray:
        cs = self.exp_data['c'].values[:17]
        n_sqr = (cs[0] + cs[1] / (x ** 2 - cs[2]) +
                 cs[3] * (x - cs[4]) / ((x - cs[4]) ** 2 + cs[5]))
        return np.sqrt(n_sqr)
