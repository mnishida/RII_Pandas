#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections.abc import Sequence
from itertools import islice
from logging import getLogger
from typing import Callable, Dict, Tuple, Union

import numpy as np
from pandas import DataFrame, Series
from scipy import constants
from scipy.interpolate import interp1d
from scipy.special import wofz

logger = getLogger(__package__)


class Material:
    """This class provide dispersion formula defined
    in refractiveindex.info database.

    Attributes:
        exp_data: The experimental data set.
        formulas (dict[int, Callable]): A dict of functions for the formulas.
    """

    def __init__(
        self,
        idx: int,
        catalog: DataFrame,
        exp_data: DataFrame,
        bound_check: bool = True,
    ):
        """Initialize Material

        Args:
            idx: ID number.
            catalog: The catalog data set.
            exp_data: The experimental data set.
            bound_check: True if bound check should be done,
        """
        self.catalog: Series = catalog.loc[idx]
        self.id = idx
        # exp_data becomes a Series if it has only 1 row.
        self.exp_data: Union[Series, DataFrame] = exp_data.loc[idx]
        self.unit = constants.h * constants.c * 1e6 / constants.e
        self.formulas: Dict[
            int,
            Callable[
                [Union[float, complex, np.ndarray]], Union[float, complex, np.ndarray]
            ],
        ] = {
            1: self.formula_1,
            2: self.formula_2,
            3: self.formula_3,
            4: self.formula_4,
            5: self.formula_5,
            6: self.formula_6,
            7: self.formula_7,
            8: self.formula_8,
            9: self.formula_9,
            21: self.formula_21,
            22: self.formula_22,
        }
        self.bound_check_flag = bound_check
        self.__n = self.func_n()
        self.__k = self.func_k()

    def bound_check(self, x: Union[float, complex, np.ndarray], nk: str):
        if not self.bound_check_flag:
            return
        if nk == "n":
            wl_min = self.catalog["wl_n_min"]
            wl_max = self.catalog["wl_n_max"]
        elif nk == "k":
            wl_min = self.catalog["wl_k_min"]
            wl_max = self.catalog["wl_k_max"]
        elif nk == "nk":
            wl_min = self.catalog["wl_min"]
            wl_max = self.catalog["wl_max"]
        else:
            raise ValueError("nk must be 'n', 'k', or 'nk'.")
        if isinstance(x, (Sequence, np.ndarray)):
            x_min = min(x)
            x_max = max(x)
        else:
            x_min = x_max = x
        if x_min < wl_min * 0.999 or x_max > wl_max * 1.001:
            raise ValueError(
                f"Wavelength [{x_min} {x_max}] is out of bounds [{wl_min} {wl_max}][um]"
            )

    def func_n(self):
        formula = int(self.catalog["formula"])
        tabulated = self.catalog["tabulated"]
        if formula > 0:
            if formula <= 20:
                return self.formulas[formula]
            else:
                return lambda x: np.sqrt(self.formulas[formula](x)).real
        elif "n" in tabulated:
            num_n = self.catalog["num_n"]
            if num_n == 1:
                return lambda x: self.exp_data["n"] * np.ones_like(x)
            else:
                return interp1d(
                    self.exp_data["wl_n"].to_numpy()[:num_n],
                    self.exp_data["n"].to_numpy()[:num_n],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
        else:
            return lambda x: np.nan * np.ones_like(x)

    def func_k(self):
        tabulated = self.catalog["tabulated"]
        if "k" in tabulated:
            num_k = self.catalog["num_k"]
            if num_k == 1:
                return lambda x: self.exp_data["k"] * np.ones_like(x)
            else:
                return interp1d(
                    self.exp_data["wl_k"].to_numpy()[:num_k],
                    self.exp_data["k"].to_numpy()[:num_k],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
        else:
            formula = int(self.catalog["formula"])
            if formula > 20:
                return lambda x: np.sqrt(self.formulas[formula](x)).imag
            else:
                return lambda x: np.nan * np.ones_like(x)

    def n(self, x: Union[float, complex, np.ndarray]) -> Union[float, np.ndarray]:
        """Return n for given wavelength

        Args:
            x: Wavelength.
        """
        self.bound_check(x.real, "n")
        return self.__n(x)

    def k(self, x: Union[float, complex, np.ndarray]) -> Union[float, np.ndarray]:
        """Return k for given wavelength

        Args:
            x: Wavelength.
        """
        self.bound_check(x.real, "k")
        return self.__k(x)

    def eps(self, x: Union[float, complex, np.ndarray]) -> Union[complex, np.ndarray]:
        """Return complex dielectric constant

        Args:
            x: Wavelength.
        """
        self.bound_check(x.real, "nk")
        formula = int(self.catalog["formula"])
        if formula > 20:
            return self.formulas[formula](x)
        else:
            n = self.__n(x.real)
            k = self.__k(x.real)
            eps = np.zeros_like(x, dtype=complex)
            eps.real = n ** 2 - k ** 2
            eps.imag = 2 * n * k
        return eps

    def plot(
        self,
        wls: Union[Sequence, np.ndarray],
        comp: str = "n",
        fmt: Union[str, None] = "-",
        **kwargs,
    ):
        """Plot refractive index.

        Args:
            wls: Array of wavelength.
            comp: 'n', 'k' or 'eps'
            fmt: The plot format string.
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault("alpha", 0.5)
        kwargs.setdefault("lw", 4)
        kwargs.setdefault("ms", 8)
        if comp == "n":
            ns = self.n(wls)
            plt.plot(wls, ns, fmt, label="{}".format(self.catalog["page"]), **kwargs)
            plt.ylabel(r"$n$")
        elif comp == "k":
            ks = self.k(wls)
            plt.plot(wls, ks, fmt, label="{}".format(self.catalog["page"]), **kwargs)
            plt.ylabel(r"$k$")
        elif comp == "eps":
            eps = self.eps(wls)
            (line,) = plt.plot(
                wls, eps.real, fmt, label="{}".format(self.catalog["page"]), **kwargs
            )
            kwargs.setdefault("color", line.get_color())
            plt.plot(wls, eps.imag, fmt, **kwargs)
            plt.ylabel(r"$\varepsilon$")
        plt.xlim(min(wls), max(wls))
        plt.xlabel(r"$\lambda$ $[\mathrm{\mu m}]$")
        plt.legend(loc="best")

    def formula_1(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_sqr = x ** 2
        n_sqr = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x_sqr / (x_sqr - c2 ** 2)
        return np.sqrt(n_sqr * (n_sqr > 0))

    def formula_2(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_sqr = x ** 2
        n_sqr = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x_sqr / (x_sqr - c2)
        return np.sqrt(n_sqr * (n_sqr > 0))

    def formula_3(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        n_sqr = cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x ** c2
        return np.sqrt(n_sqr * (n_sqr > 0))

    def formula_4(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        n_sqr = (
            cs[0]
            + cs[1] * x ** cs[2] / (x ** 2 - cs[3] ** cs[4])
            + cs[5] * x ** cs[6] / (x ** 2 - cs[7] ** cs[8])
        )
        for c1, c2 in zip(islice(cs, 9, None, 2), islice(cs, 10, None, 2)):
            n_sqr += c1 * x ** c2
        return np.sqrt(n_sqr * (n_sqr > 0))

    def formula_5(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        n = cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n += c1 * x ** c2
        return n

    def formula_6(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_m2 = 1 / x ** 2
        n = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n += c1 / (c2 - x_m2)
        return n

    def formula_7(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
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

    def formula_8(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_sqr = x ** 2
        a = cs[0] + cs[1] * x_sqr / (x_sqr - cs[2]) + cs[3] * x_sqr
        n_sqr = (1 + 2 * a) / (1 - a)
        return np.sqrt(n_sqr * (n_sqr > 0))

    def formula_9(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        cs = self.exp_data["c"].to_numpy()[:24]
        n_sqr = (
            cs[0]
            + cs[1] / (x ** 2 - cs[2])
            + cs[3] * (x - cs[4]) / ((x - cs[4]) ** 2 + cs[5])
        )
        return np.sqrt(n_sqr * (n_sqr > 0))

    def formula_21(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        cs = self.exp_data["c"].to_numpy()[:24]
        w = self.unit / x
        eb = cs[0]
        f0 = cs[1]
        g0 = cs[2]
        wp = cs[3]
        eps = eb - f0 * wp ** 2 / (w ** 2 + 1j * w * g0)
        for fj, gj, wj in zip(
            islice(cs, 4, None, 3), islice(cs, 5, None, 3), islice(cs, 6, None, 3)
        ):
            eps -= fj * wp ** 2 / (w ** 2 - wj ** 2 + 1j * w * gj)
        return eps

    def formula_22(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        cs = self.exp_data["c"].to_numpy()[:24]
        w = self.unit / x
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
