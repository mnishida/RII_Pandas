#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from itertools import islice
from logging import getLogger
from re import A, M
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy import constants
from scipy.interpolate import interp1d
from scipy.special import wofz

import riip.dataframe

logger = getLogger(__package__)


def _ensure_positive_imag(x: ArrayLike) -> ArrayLike:
    return x.real + 1j * x.imag * (x.imag > 0)


class AbstractMaterial(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self.label = ""

    @abc.abstractmethod
    def n(self, wls: ArrayLike) -> ArrayLike:
        pass

    @abc.abstractmethod
    def k(self, wls: ArrayLike) -> ArrayLike:
        pass

    @abc.abstractmethod
    def eps(self, wls: ArrayLike) -> ArrayLike:
        pass

    def plot(
        self,
        wls: Sequence | np.ndarray,
        comp: str = "n",
        fmt1: Optional[str] = "-",
        fmt2: Optional[str] = "--",
        **kwargs,
    ):
        """Plot refractive index, extinction coefficient or permittivity.

        Args:
            wls (Union[Sequence, np.ndarray]): Wavelength coordinates to be plotted [μm].
            comp (str): 'n', 'k' or 'eps'
            fmt1 (Union[str, None]): Plot format for n and Re(eps).
            fmt2 (Union[str, None]): Plot format for k and Im(eps).
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault("lw", 4)
        kwargs.setdefault("ms", 8)
        if comp == "n":
            ns = self.n(wls)
            plt.plot(wls, ns, fmt1, label=self.label, **kwargs)
            plt.ylabel(r"$n$")
        elif comp == "k":
            ks = self.k(wls)
            plt.plot(wls, ks, fmt2, label=self.label, **kwargs)
            plt.ylabel(r"$k$")
        elif comp == "eps":
            eps = self.eps(wls)
            (line,) = plt.plot(wls, eps.real, fmt1, label=self.label, **kwargs)
            color = line.get_color()
            plt.plot(wls, eps.imag, fmt2, color=color, **kwargs)
            plt.ylabel(r"$\varepsilon$")
        plt.xlabel(r"$\lambda$ $[\mathrm{\mu m}]$")
        plt.legend()


class RiiMaterial(AbstractMaterial):
    """This class provide dispersion formula defined in refractiveindex.info database.

    Attributes:
        exp_data: The experimental data set.
        formulas (dict[int, Callable]): A dict of functions for the formulas.
    """

    def __init__(
        self,
        idx: int,
        catalog: DataFrame,
        raw_data: DataFrame,
        bound_check: bool = True,
    ) -> None:
        """Initialize RiiMaterial

        Args:
            idx (int): 'id' in RiiDataFrame.catalog.
            catalog (DataFrame): 'RiiDataFrame.catalog.
            raw_data (DataFrame): 'RiiDataFrame.raw_data.
            bound_check: True if bound check should be done.
        """
        self.catalog: Series = catalog.loc[idx]
        # exp_data becomes a Series if it has only 1 row.
        self.exp_data: Series | DataFrame = raw_data.loc[idx]
        self.label = self.catalog["page"]
        self.unit = constants.h * constants.c * 1e6 / constants.e
        self.formulas: dict[int, Callable] = {
            1: self._formula_1,
            2: self._formula_2,
            3: self._formula_3,
            4: self._formula_4,
            5: self._formula_5,
            6: self._formula_6,
            7: self._formula_7,
            8: self._formula_8,
            9: self._formula_9,
            21: self._formula_21,
            22: self._formula_22,
        }
        self.bound_check_flag = bound_check
        self.__n = self._func_n()
        self.__k = self._func_k()

    def _bound_check(self, x: ArrayLike, nk: str) -> None:
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
            _x = np.asarray(x).real
            x_min = min(_x)
            x_max = max(_x)
        else:
            x_min = x_max = x.real
        if x_min < wl_min or x_max > wl_max:
            raise ValueError(
                f"Wavelength [{x_min} {x_max}] is out of bounds [{wl_min} {wl_max}][um]"
            )

    def _func_n(self) -> Callable:
        formula = int(self.catalog["formula"])
        tabulated = self.catalog["tabulated"]
        if formula > 0:
            if formula <= 20:
                return self.formulas[formula]
            else:
                return lambda x: np.sqrt(
                    _ensure_positive_imag(self.formulas[formula](x))
                ).real
        elif "n" in tabulated:
            num_n = self.catalog["num_n"]
            if num_n == 1:
                return lambda x: self.exp_data["n"] * np.ones_like(x)
            else:
                return interp1d(
                    self.exp_data["wl_n"].to_numpy()[:num_n],
                    self.exp_data["n"].to_numpy()[:num_n],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
        else:
            logger.warning("Refractive index is missing and set to zero.")
            return lambda x: np.zeros_like(x)

    def _func_k(self) -> Callable:
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
                return lambda x: np.sqrt(
                    _ensure_positive_imag(self.formulas[formula](x))
                ).imag
            else:
                logger.warning("Extinction index is missing and set to zero.")
                return lambda x: np.zeros_like(x)

    def n(self, wl: ArrayLike) -> ArrayLike:
        """Return refractive index at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        self._bound_check(wl, "n")
        return self.__n(np.asarray(wl).real)

    def k(self, wl: ArrayLike) -> ArrayLike:
        """Return extinction coefficient at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        self._bound_check(wl, "k")
        return self.__k(np.asarray(wl).real)

    def eps(self, wl: ArrayLike) -> ArrayLike:
        """Return complex dielectric constant at given wavelength.

        Args:
            wl (Union[float, complex, Sequence, np.ndarray]): Wavelength [μm].
        """
        formula = int(self.catalog["formula"])
        if formula > 20:
            self._bound_check(wl, "nk")
            return self.formulas[formula](np.asarray(wl))
        _wl = np.asarray(wl)
        n: np.ndarray = self.n(_wl)
        k: np.ndarray = self.k(_wl)
        eps = n ** 2 - k ** 2 + 2j * n * k
        return eps

    def _formula_1(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_sqr = x ** 2
        n_sqr = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x_sqr / (x_sqr - c2 ** 2)
        return np.sqrt(n_sqr * (n_sqr > 0))

    def _formula_2(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_sqr = x ** 2
        n_sqr = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x_sqr / (x_sqr - c2)
        return np.sqrt(n_sqr * (n_sqr > 0))

    def _formula_3(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        n_sqr = cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n_sqr += c1 * x ** c2
        return np.sqrt(n_sqr * (n_sqr > 0))

    def _formula_4(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        n_sqr = (
            cs[0]
            + cs[1] * x ** cs[2] / (x ** 2 - cs[3] ** cs[4])
            + cs[5] * x ** cs[6] / (x ** 2 - cs[7] ** cs[8])
        )
        for c1, c2 in zip(islice(cs, 9, None, 2), islice(cs, 10, None, 2)):
            n_sqr += c1 * x ** c2
        return np.sqrt(n_sqr * (n_sqr > 0))

    def _formula_5(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        n = cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n += c1 * x ** c2
        return n

    def _formula_6(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_m2 = 1 / x ** 2
        n = 1 + cs[0]
        for c1, c2 in zip(islice(cs, 1, None, 2), islice(cs, 2, None, 2)):
            n += c1 / (c2 - x_m2)
        return n

    def _formula_7(self, x: ArrayLike) -> ArrayLike:
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

    def _formula_8(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        x_sqr = x ** 2
        a = cs[0] + cs[1] * x_sqr / (x_sqr - cs[2]) + cs[3] * x_sqr
        n_sqr = (1 + 2 * a) / (1 - a)
        return np.sqrt(n_sqr * (n_sqr > 0))

    def _formula_9(self, x: ArrayLike) -> ArrayLike:
        cs = self.exp_data["c"].to_numpy()[:24]
        n_sqr = (
            cs[0]
            + cs[1] / (x ** 2 - cs[2])
            + cs[3] * (x - cs[4]) / ((x - cs[4]) ** 2 + cs[5])
        )
        return np.sqrt(n_sqr * (n_sqr > 0))

    def _formula_21(self, x: ArrayLike) -> ArrayLike:
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

    def _formula_22(self, x: ArrayLike) -> ArrayLike:
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


class Material(AbstractMaterial):
    def __init__(self, params: dict) -> None:
        """Initialize Material

        Args:
            params (Dict): parameter dict contains the following key and values
                'book': book value in catalog of RiiDataFrame. (str)
                'page': page value in catalog of RiiDataFrame. (str)
                'RI': Constant refractive index. (complex)
                'e': Constant permittivity. (complex)
                'bound_check': True if bound check should be done. Defaults to True. (bool)
                'im_factor': A magnification factor multiplied to the imaginary part of permittivity. Defaults to 1.0. (float)
        """
        self.rim: Optional[RiiMaterial] = None
        self.__ce0: Optional[complex] = None
        self.__ce: Optional[complex] = None
        self.__cn: Optional[float] = None
        self.__ck: Optional[float] = None
        if "RI" in params:
            self.__ce0 = params["RI"] ** 2
            self.__label = f"RI: {params['RI']}"
            if "e" in params:
                if params["e"] != params["RI"] ** 2:
                    raise ValueError("e must be RI ** 2.")
        elif "e" in params:
            self.__label = r"$\varepsilon$" + f": {params['e']}"
            self.__ce0 = params["e"]
        elif "book" not in params or "page" not in params:
            raise ValueError("'RI', 'e' or ('book' and 'page') must be specified")
        else:
            rii = riip.dataframe.RiiDataFrame()
            book = params["book"]
            page = params["page"]
            idx = rii.catalog.query(f"book == '{book}' and page == '{page}'").index[0]
            self.__label = page
            self.rim = RiiMaterial(
                idx, rii.catalog, rii.raw_data, params.get("bound_check", None)
            )
        self.im_factor = params.get("im_factor", 1.0)

    def _set_constants(self, im_factor: float) -> None:
        if im_factor != 1.0:
            self.label = self.__label + f" im_factor: {self.__im_factor}"
        else:
            self.label = self.__label
        if self.__ce0:
            imag = self.__ce0.imag * im_factor
            self.__ce = self.__ce0.real + 1j * imag * (imag > 0)
            _ri = 1j * np.sqrt(self.__ce)
            self.__cn = _ri.real
            self.__ck = _ri.imag

    @property
    def im_factor(self) -> float:
        return self.__im_factor

    @im_factor.setter
    def im_factor(self, factor: float) -> None:
        self.__w = None
        self.__im_factor = factor
        self._set_constants(factor)

    def n(self, wl: float | complex) -> float:
        """Return refractive index at given wavelength.

        Args:
            wl (float | complex): Wavelength [μm].
        """
        if self.__cn is None:
            return self.rim.n(wl)
        if isinstance(wl, (Sequence, np.ndarray)):
            return self.__cn * np.ones_like(wl, dtype=float)
        return self.__cn

    def k(self, wl: float | complex) -> float:
        """Return extinction coefficient at given wavelength.

        Args:
            wl (float | complex): Wavelength [μm].
        """
        if self.__ck is None:
            return self.rim.k(wl)
        if isinstance(wl, (Sequence, np.ndarray)):
            return self.__ck * np.ones_like(wl, dtype=float)
        return self.__ck

    def eps(self, wl: float | complex) -> complex:
        """Return complex dielectric constant at given wavelength.

        Args:
            wl (float | complex): Wavelength [μm].
        """
        if self.__ce is None:
            e = self.rim.eps(wl)
            imag = e.imag * self.im_factor
            return e.real + 1j * imag
        if isinstance(wl, (Sequence, np.ndarray)):
            return self.__ce * np.ones_like(wl, dtype=float)
        return self.__ce

    def __call__(self, w: float | complex) -> complex:
        """Return relative permittivity at given angular frequency.

        Args:
            w (float | complex): A float indicating the angular frequency.

        Returns:
            complex: Relative permittivity at w

        Raises:
            ValueError: The model is not defined.
        """
        if self.__w is None or w != self.__w:
            self.__w = w
            self.__e = self.eps(2 * np.pi / w)
        return self.__e
