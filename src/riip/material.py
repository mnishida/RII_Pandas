from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from logging import getLogger
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.interpolate import interp1d
from scipy.special import wofz

import riip.dataframe

from .formulas import formulas_cython_dict, formulas_numpy_dict

logger = getLogger(__package__)


def _ensure_positive_imag(x: ArrayLike) -> np.ndarray:
    """If the imaginary part of x is negative, change it to zero."""
    _x = np.asarray(x, dtype=np.complex128)
    return _x.real + 1j * _x.imag * (_x.imag > 0)


class AbstractMaterial(metaclass=abc.ABCMeta):
    """Abstract base class for materials"""

    @abc.abstractmethod
    def __init__(self, *args) -> None:
        self.label = ""

    @abc.abstractmethod
    def n(self, wls: ArrayLike) -> np.ndarray:
        """Retrun refractive index at wavelength wls [μm]"""
        return np.asarray(wls, dtype=np.float64)

    @abc.abstractmethod
    def k(self, wls: ArrayLike) -> np.ndarray:
        """Return extinction coefficient at wavelength wls [μm]"""
        return np.asarray(wls, dtype=np.float64)

    @abc.abstractmethod
    def eps(self, wls: ArrayLike) -> np.ndarray:
        """Return permittivity at wavelength wls [μm]"""
        return np.asarray(wls, dtype=np.float64)

    def bound_check(self, wl: ArrayLike, nk: str) -> None:
        pass

    def plot(
        self,
        wls: Sequence | np.ndarray,
        comp: str = "n",
        fmt1: Optional[str] = "-",
        fmt2: Optional[str] = "--",
        **kwargs,
    ) -> None:
        """Plot refractive index, extinction coefficient or permittivity.

        Args:
            wls (Sequence | np.ndarray): Wavelength coordinates to be plotted [μm].
            comp (str): 'n', 'k' or 'eps'
            fmt1 (Optional[str]): Plot format for n and Re(eps).
            fmt2 (Optional[str]): Plot format for k and Im(eps).
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
        catalog: Catalog of data.
        raw_data: The experimental data set.
    """

    def __init__(
        self, id: int, catalog: DataFrame, raw_data: DataFrame, bound_check: bool = True
    ) -> None:
        """Initialize RiiMaterial

        Args:
            id (int): ID number
            catalog (DataFrame): catalog of Rii_Pandas DataFrame.
            raw_data (DataFrame): raw_data of Rii_Pandas DataFrame.
            bound_check (bool): True if bound check should be done. Defaults to True.
        """
        self.catalog: Series = catalog.loc[id]
        # raw_data becomes a Series if it has only 1 row.
        self.raw_data: Series | DataFrame = raw_data.loc[id]
        self.f = int(self.catalog["formula"])
        if self.f > 0:
            self.cs = self.raw_data["c"].to_numpy()[:24]
            self.formula = lambda x: formulas_numpy_dict[self.f](x, self.cs)
        self.label = self.catalog["page"]
        self.bound_check_flag = bound_check
        self.__n = self._func_n()
        self.__k = self._func_k()

    def bound_check(self, wl: ArrayLike, nk: str) -> None:
        """Raise ValueError if wl is out of bounds"""
        _x = np.atleast_1d(wl)
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
        x_min = min(_x)
        x_max = max(_x)
        if x_min < wl_min or x_max > wl_max:
            raise ValueError(
                f"Wavelength [{x_min} {x_max}] is out of bounds [{wl_min} {wl_max}][um]"
            )

    def _func_n(self) -> Callable:
        tabulated = self.catalog["tabulated"]
        if self.f > 0:
            if self.f <= 20:
                return self.formula
            else:
                return lambda x: np.sqrt(_ensure_positive_imag(self.formula(x))).real
        elif "n" in tabulated:
            num_n = self.catalog["num_n"]
            if num_n == 1:
                return lambda x: self.raw_data["n"] * np.ones_like(x)
            elif num_n < 4:
                val = np.mean(self.raw_data["n"])
                return lambda x: val * np.ones_like(x)
            else:
                return interp1d(
                    self.raw_data["wl_n"].to_numpy()[:num_n],
                    self.raw_data["n"].to_numpy()[:num_n],
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
                return lambda x: self.raw_data["k"] * np.ones_like(x)
            elif num_k < 4:
                val = np.mean(self.raw_data["k"])
                return lambda x: val * np.ones_like(x)
            else:
                return interp1d(
                    self.raw_data["wl_k"].to_numpy()[:num_k],
                    self.raw_data["k"].to_numpy()[:num_k],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
        else:
            formula = int(self.catalog["formula"])
            if formula > 20:
                return lambda x: np.sqrt(_ensure_positive_imag(self.formula(x))).imag
            else:
                logger.warning("Extinction coefficient is missing and set to zero.")
                return lambda x: np.zeros_like(x)

    def n(self, wl: ArrayLike) -> np.ndarray:
        """Return refractive index at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        _wl = np.asarray(wl)
        self.bound_check(_wl, "n")
        return self.__n(_wl)

    def k(self, wl: ArrayLike) -> np.ndarray:
        """Return extinction coefficient at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        _wl = np.asarray(wl)
        self.bound_check(_wl, "k")
        return self.__k(_wl)

    def eps(self, wl: ArrayLike) -> np.ndarray:
        """Return complex dielectric constant at given wavelength.

        Args:
            wl (Union[float, complex, Sequence, np.ndarray]): Wavelength [μm].
        """
        _wl = np.asarray(wl)
        if self.f > 20:
            self.bound_check(_wl, "nk")
            return self.formula(_wl)
        n: np.ndarray = self.n(_wl)
        k: np.ndarray = self.k(_wl)
        eps = n ** 2 - k ** 2 + 2j * n * k
        return eps


class ConstMaterial(AbstractMaterial):
    """A class defines a material with constant permittivity

    Attributes:
        ce (complex): The value of constant permittivity
        label (str): A label used in plot
    """

    def __init__(self, params: dict) -> None:
        """Initialize Material

        Args:
            params (Dict): parameter dict contains the following key and values
                'RI': Constant refractive index. (complex)
                'e': Constant permittivity. (complex)
            rid (RiiDataFrame): Rii_Pandas DataFrame.
        """
        if "RI" in params:
            RI = params["RI"]
            self.ce = RI ** 2 + 0.0j
            self.cn = RI.real
            self.ck = RI.imag
            self.label = f"RI: {RI}"
            if "e" in params:
                e = params["e"]
                if e != RI ** 2:
                    raise ValueError("e must be RI ** 2.")
        elif "e" in params:
            e = params["e"]
            self.label = r"$\varepsilon$" + f": {e}"
            self.ce = e + 0.0j
            ri = np.sqrt(_ensure_positive_imag(e))
            self.cn = ri.real
            self.ck = ri.imag
        else:
            raise ValueError("'RI' or 'e' must be specified")

    def n(self, wl: ArrayLike) -> np.ndarray:
        """Return refractive index at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        return self.cn * np.ones_like(wl)

    def k(self, wl: ArrayLike) -> np.ndarray:
        """Return extinction coefficient at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        return self.ck * np.ones_like(wl)

    def eps(self, wl: ArrayLike) -> np.ndarray:
        """Return complex dielectric constant at given wavelength.

        Args:
            wl (Union[float, complex, Sequence, np.ndarray]): Wavelength [μm].
        """
        return self.ce * np.ones_like(wl)


class Material(AbstractMaterial):
    """A Class that constructs RiiMaterial or ConstMaterial instance depending on the given parameters.

    Implement __call__ method that calculate the permittivity at a single value of angular frequency.
    Introduce 'im_factor' that is a magnification factor multiplied to the imaginary part of permittivity.

    Args:
        AbstractMaterial ([type]): [description]
    """

    def __init__(
        self, params: dict, rid: Optional[riip.dataframe.RiiDataFrame] = None
    ) -> None:
        """Initialize Material

        Args:
            params (dict): parameter dict contains the following key and values
                'id': ID number (int)
                'book': book value in catalog of RiiDataFrame. (str)
                'page': page value in catalog of RiiDataFrame. (str)
                'RI': Constant refractive index. (complex)
                'e': Constant permittivity. (complex)
                'bound_check': True if bound check should be done. Defaults to True. (bool)
                'im_factor': A magnification factor multiplied to the imaginary part of permittivity. Defaults to 1.0. (float)
            rid (RiiDataFrame): Rii_Pandas DataFrame. Defaults to None.
        """
        if "RI" in params or "e" in params:
            self.material: ConstMaterial | RiiMaterial = ConstMaterial(params)
            self.__ce0: Optional[complex] = self.material.ce
            self.f = 0
        elif "id" not in params and ("book" not in params or "page" not in params):
            raise ValueError("'RI', 'e', 'id', or 'book'-'page' pair must be specified")
        else:
            if rid is None:
                rid = riip.dataframe.RiiDataFrame()
            if "book" in params and "page" in params:
                idx = rid.book_page_to_id(params)
                if "id" in "params":
                    idx != params["id"]
                    raise ValueError(
                        "There is an inconsistency between 'id' and 'book'-'page' pair"
                    )
                else:
                    params["id"] = idx
            self.material = RiiMaterial(
                params["id"], rid.catalog, rid.raw_data, params.get("bound_check", True)
            )
            self.catalog = self.material.catalog
            self.wl_max = self.catalog["wl_max"]
            self.wl_min = self.catalog["wl_min"]
            self.raw_data = self.material.raw_data
            self.bound_check_flag = self.material.bound_check_flag
            self.__ce0 = None
            self.f = self.material.f
            if self.f != 0:
                self.cs = self.material.cs
        self.__w: Optional[float | complex] = None
        self.__ce: Optional[complex] = self.__ce0
        self.im_factor = params.get("im_factor", 1.0)

    def eps(self, wl: ArrayLike) -> np.ndarray:
        """Return complex dielectric constant at given wavelength.

        Args:
            wl (Union[float, complex, Sequence, np.ndarray]): Wavelength [μm].
        """
        e = self.material.eps(wl)
        if self.__im_factor != 1.0:
            imag = e.imag * self.__im_factor
            e = e.real + 1j * imag
        return e

    def n(self, wl: ArrayLike) -> np.ndarray:
        """Return refractive index at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        return np.sqrt(_ensure_positive_imag(self.eps(wl))).real

    def k(self, wl: ArrayLike) -> np.ndarray:
        """Return extinction coefficient at given wavelength.

        Args:
            wl (ArrayLike): Wavelength [μm].
        """
        return np.sqrt(_ensure_positive_imag(self.eps(wl))).imag

    @property
    def im_factor(self) -> float:
        return self.__im_factor

    @im_factor.setter
    def im_factor(self, factor: float) -> None:
        self.__w = None
        self.__im_factor = factor
        if factor != 1.0:
            self.label = self.material.label + f" im_factor: {factor}"
            if self.__ce0 is not None:
                if factor != 1.0:
                    imag = self.__ce0 * factor
                    self.__ce = self.__ce0.real + 1j * imag
        else:
            self.label = self.material.label

    def __call__(self, w: float | complex) -> complex:
        """Return relative permittivity at given angular frequency.

        Args:
            w (float | complex): A float indicating the angular frequency (vacuum wavenumber ω/c [rad/μm]).

        Returns:
            complex: Relative permittivity at w
        """
        if self.__ce is not None:
            return self.__ce
        if self.__w is None or w != self.__w:
            wl = 2 * np.pi / w.real
            if wl < self.wl_min or wl > self.wl_max:
                raise ValueError(
                    f"Wavelength {wl} is out of bounds [{self.wl_min} {self.wl_max}][um]"
                )
            if self.f > 0:
                if self.f > 20:
                    self.__e = formulas_cython_dict[self.f](w, self.cs)
                else:
                    _n = formulas_cython_dict[self.f](w.real, self.cs)
                    _k = self.material.k(2 * np.pi / w.real).item()
                    self.__e = _n ** 2 - _k ** 2 + 2j * _n * _k
                if self.__im_factor != 1.0:
                    imag = self.__e.imag * self.__im_factor
                    self.__e = self.__e.real + 1j * imag
            else:
                self.__e = self.eps(wl).item()
            self.__w = w
        return self.__e
