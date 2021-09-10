#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from logging import ERROR, WARNING, getLogger
from typing import ClassVar, Optional

import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import HTML
from pandas import DataFrame
from pandas.core.series import Series

import riip.material

# from numpy.typing import ArrayLike


logger = getLogger(__package__)
_dirname = os.path.dirname(__file__)
_ri_database = os.path.join(_dirname, "data", "refractiveindex.info-database")
_db_directory = os.path.join(_ri_database, "database")
_my_db_directory = os.path.join(_dirname, "data", "my_database")
_catalog_file = os.path.join(_dirname, "data", "catalog.csv")
_raw_data_file = os.path.join(_dirname, "data", "raw_data.csv")
_grid_data_file = os.path.join(_dirname, "data", "grid_data.h5")
_ri_database_repo = (
    "https://github.com/polyanskiy/" + "refractiveindex.info-database.git"
)
_ri_database_patch = os.path.join(_dirname, "data", "riid.patch")


class RiiDataFrame:
    """A class that provides a Pandas DataFrame for 'refractiveindex.info database'.

    Attributes:
        db_path: The path to the refractiveindex.info-database/database.
        my_db_path: The path to my_database.
        catalog: The catalog.
        catalog_file: The csv filename to store the catalog.
        raw_data: The experimental data.
        raw_data_file: The csv filename to store the raw_data.
        grid_data_file: The hdf5 filename to store the grid_data.

    """

    _catalog_columns: ClassVar[OrderedDict] = OrderedDict(
        (
            ("id", np.int32),
            ("shelf", str),
            ("shelf_name", str),
            ("division", str),
            ("book", str),
            ("book_name", str),
            ("section", str),
            ("page", str),
            ("path", str),
            ("formula", np.int32),
            ("tabulated", str),
            ("num_n", np.int32),
            ("num_k", np.int32),
            ("wl_n_min", np.float64),
            ("wl_n_max", np.float64),
            ("wl_k_min", np.float64),
            ("wl_k_max", np.float64),
            ("wl_min", np.float64),
            ("wl_max", np.float64),
        )
    )

    _raw_data_columns: ClassVar[OrderedDict] = OrderedDict(
        (
            ("id", np.int32),
            ("c", np.float64),
            ("wl_n", np.float64),
            ("n", np.float64),
            ("wl_k", np.float64),
            ("k", np.float64),
        )
    )

    _grid_data_columns: ClassVar[OrderedDict] = OrderedDict(
        (("id", np.int32), ("wl", np.float64), ("n", np.float64), ("k", np.float64))
    )

    def __init__(
        self,
        db_path: str = _db_directory,
        catalog_file: str = _catalog_file,
        raw_data_file: str = _raw_data_file,
        grid_data_file: str = _grid_data_file,
        my_db_path: str = _my_db_directory,
    ):
        """Initialize the RiiDataFrame.

        Args:
            db_path: The path to the refractiveindex.info-database/database.
            my_db_path: The path to my_database.
            catalog_file: The filename of the catalog csv file.
            raw_data_file: The filename of the experimental data csv file.
            grid_data_file: The filename of the grid wl-nk data csv file.
        """
        self._db_path: str = db_path
        self._my_db_path: str = my_db_path
        self._ri_database: str = os.path.dirname(self._db_path)
        self._catalog_file: str = catalog_file
        self._raw_data_file: str = raw_data_file
        self._grid_data_file: str = grid_data_file
        _catalog, _raw_data = self._load_catalog_and_raw_data()
        self.catalog: DataFrame = _catalog
        self.raw_data: DataFrame = _raw_data
        self.__book_page_order = self._create_book_page_order()

    def _load_catalog_and_raw_data(self) -> tuple[DataFrame, DataFrame]:
        # Create csv files
        if not os.path.isfile(self._catalog_file):
            logger.warning("Catalog file not found.")
            if not os.path.isfile(os.path.join(self._db_path, "library.yml")):
                logger.warning("Cloning Repository...")
                repo = git.Repo.clone_from(
                    _ri_database_repo, self._ri_database, branch="master"
                )
                repo.git.apply(_ri_database_patch)
                # git.Repo.clone_from(
                #     _ri_database_repo, self._ri_database, branch="master"
                # )
                logger.warning("Done.")
            logger.warning("Creating catalog file...")
            catalog = self._add_my_db_to_catalog(self._create_catalog())
            logger.warning("Done.")

            # Preparing raw_data
            logger.warning("Creating raw data file...")
            raw_data, catalog = self._create_raw_data_and_modify_catalog(catalog)
            logger.warning("Done.")

            # Preparing grid_data
            logger.warning("Updating grid data file...")
            catalog = catalog.set_index("id")
            raw_data = raw_data.set_index("id")
            self._create_grid_data(catalog, raw_data)
            logger.warning("Done.")
        else:
            catalog = pd.read_csv(
                self._catalog_file,
                dtype=self._catalog_columns,
                index_col="id",
                na_filter=False,
            )
            raw_data = pd.read_csv(
                self._raw_data_file,
                dtype=self._raw_data_columns,
                index_col="id",
                na_filter=False,
            )
        return catalog, raw_data

    @staticmethod
    def _extract_entry(db_path: str, start_id: int = 0) -> Iterable:
        """Yield a single data set."""
        reference_path = os.path.normpath(db_path)
        library_file = os.path.join(reference_path, "library.yml")
        with open(library_file, "r", encoding="utf-8") as f:
            library = yaml.safe_load(f)
        idx = start_id
        shelf = "main"
        book = "Ag (Experimental data)"
        page = "Johnson"
        try:
            for sh in library:
                shelf = sh["SHELF"]
                if shelf == "3d":
                    # This shelf does not seem to contain new data.
                    break
                shelf_name = sh["name"]
                division = None
                for b in sh["content"]:
                    if "DIVIDER" in b:
                        division = b["DIVIDER"]
                    else:
                        if division is None:
                            raise Exception("'DIVIDER' is missing in 'library.yml'.")
                        if "DIVIDER" not in b["content"]:
                            section = ""
                        for p in b["content"]:
                            if "DIVIDER" in p:
                                # This DIVIDER specifies the phase of the
                                #  material such as gas, liquid or solid, so it
                                #  is added to the book and book_name with
                                #  parentheses.
                                section = p["DIVIDER"]
                            else:
                                book = b["BOOK"]
                                book_name = b["name"]
                                page = p["PAGE"]
                                path = os.path.join(
                                    reference_path, "data", os.path.normpath(p["data"])
                                )
                                logger.debug("{0} {1} {2}".format(idx, book, page))
                                row = [idx, shelf, shelf_name, division]
                                row += [book, book_name, section, page, path]
                                row += [0, "f", 0, 0, 0, 0, 0, 0, 0, 0]
                                yield row
                                idx += 1
        except Exception as e:
            message = (
                "There seems to be some inconsistency in the library.yml "
                + "around id={}, shelf={}, book={}, page={}.".format(
                    idx, shelf, book, page
                )
            )
            raise Exception(message) from e

    def _create_catalog(self) -> DataFrame:
        """Create catalog DataFrame from library.yml."""
        logger.info("Creating catalog...")
        df = DataFrame(
            list(self._extract_entry(self._db_path)),
            columns=self._catalog_columns.keys(),
        )
        logger.info("Done.")
        return df.astype(self._catalog_columns)

    def _add_my_db_to_catalog(self, catalog: DataFrame) -> DataFrame:
        """Add data in my_database to catalog DataFrame."""
        logger.info("Adding my_db to catalog...")
        start_id = catalog["id"].values[-1] + 1
        logger.debug(start_id)
        df = DataFrame(
            list(self._extract_entry(self._my_db_path, start_id)),
            columns=self._catalog_columns.keys(),
        )
        df = catalog.append(df, ignore_index=True)
        logger.info("Done.")
        return df

    def _create_book_page_order(self) -> Series:
        """Create [id, book+page string] array used to search id."""
        cl = self.catalog
        book_page = {
            idx: f"{cl.loc[idx, 'book']}{cl.loc[idx, 'page']}" for idx in cl.index
        }
        return Series(book_page).sort_values()

    def book_page_to_id(self, params: dict) -> int:
        bp = params["book"] + params["page"]
        ind = np.searchsorted(self.__book_page_order, bp)
        return self.__book_page_order.index[ind]

    def _extract_raw_data(
        self, idx: int, catalog: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        """Yield a single raw data set.

        Some data are inserted into the catalog.

        Args:
            catalog: The catalog.
            idx: The ID number of the data set.
        """
        path = catalog.loc[idx, "path"]
        with open(path, "r", encoding="utf-8") as f:
            data_list = yaml.safe_load(f)["DATA"]
        wl_n_min = wl_k_min = 0.0
        wl_n_max = wl_k_max = np.inf
        formula = 0
        tabulated = ""
        cs = []
        wls_n = []
        wls_k = []
        ns = []
        ks = []
        num_n = num_k = 0
        for data in data_list:
            data_type, data_set = data["type"].strip().split()

            # For tabulated data
            if data_type == "tabulated":
                if data_set == "nk":
                    tabulated += data_set
                    wls_n, ns, ks = np.array(
                        [
                            line.strip().split()
                            for line in data["data"].strip().split("\n")
                        ],
                        dtype=float,
                    ).T
                    wls_n, inds = np.unique(wls_n, return_index=True)
                    ns = ns[inds]
                    ks = ks[inds]
                    inds = np.argsort(wls_n)
                    wls_n = list(wls_n[inds])
                    wls_k = wls_n
                    ns = list(ns[inds])
                    ks = list(ks[inds])
                    wl_n_min = wl_k_min = wls_n[0]
                    wl_n_max = wl_k_max = wls_n[-1]
                    num_n = len(wls_n)
                    num_k = len(wls_k)
                elif data_set == "n":
                    tabulated += data_set
                    wls_n, ns = np.array(
                        [
                            line.strip().split()
                            for line in data["data"].strip().split("\n")
                        ],
                        dtype=float,
                    ).T
                    wls_n, inds = np.unique(wls_n, return_index=True)
                    ns = ns[inds]
                    inds = np.argsort(wls_n)
                    wls_n = list(wls_n[inds])
                    ns = list(ns[inds])
                    wl_n_min = wls_n[0]
                    wl_n_max = wls_n[-1]
                    num_n = len(wls_n)
                elif data_set == "k":
                    tabulated += data_set
                    wls_k, ks = np.array(
                        [
                            line.strip().split()
                            for line in data["data"].strip().split("\n")
                        ],
                        dtype=float,
                    ).T
                    wls_k, inds = np.unique(wls_k, return_index=True)
                    ks = ks[inds]
                    inds = np.argsort(wls_k)
                    wls_k = list(wls_k[inds])
                    ks = list(ks[inds])
                    wl_k_min = wls_k[0]
                    wl_k_max = wls_k[-1]
                    num_k = len(wls_k)
                else:
                    raise Exception("DATA is broken.")
            # For formulas
            elif data_type == "formula":
                formula = data_set
                wl_n_min, wl_n_max = [
                    float(s) for s in data["wavelength_range"].strip().split()
                ]
                cs = [float(s) for s in data["coefficients"].strip().split()]
            else:
                raise Exception("DATA has unknown contents {}".format(data_type))

        if len(tabulated) > 2:
            raise Exception("Too many tabulated data set are provided")
        elif "nn" in tabulated or "kk" in tabulated:
            raise Exception("There is redundancy in n or k.")
        elif tabulated == "kn":
            tabulated = "nk"
        elif tabulated == "":
            tabulated = "f"

        if tabulated == "k" and formula != 0:
            if wl_n_min < wl_k_min:
                wls_k = [wl_n_min] + wls_k
                ks = [min(ks)] + ks
                num_k += 1
            if wl_n_max > wl_k_max:
                wls_k = wls_k + [wl_n_max]
                ks = ks + [min(ks)]
                num_k += 1
            wl_k_min, wl_k_max = wl_n_min, wl_n_max

        if "k" not in tabulated:
            wl_k_min, wl_k_max = wl_n_min, wl_n_max

        wl_min = max(wl_n_min, wl_k_min)
        wl_max = min(wl_n_max, wl_k_max)

        # The coefficients not included in the formula must be zero.
        num_c = len(cs)
        if formula != 0:
            cs += [0.0] * (24 - num_c)
            num_c = 24

        # All the arrays must have the same length.
        num = max(num_n, num_k, num_c)
        _cs = np.array(cs + [0.0] * (num - num_c), dtype=np.float64)
        _wls_n = np.array(wls_n + [0.0] * (num - num_n), dtype=np.float64)
        _ns = np.array(ns + [0.0] * (num - num_n), dtype=np.float64)
        _wls_k = np.array(wls_k + [0.0] * (num - num_k), dtype=np.float64)
        _ks = np.array(ks + [0.0] * (num - num_k), dtype=np.float64)

        # Rewrite catalog with the obtained data
        catalog.loc[idx, "formula"] = formula
        catalog.loc[idx, "tabulated"] = tabulated
        catalog.loc[idx, "num_n"] = num_n
        catalog.loc[idx, "num_k"] = num_k
        catalog.loc[idx, "wl_n_min"] = wl_n_min
        catalog.loc[idx, "wl_n_max"] = wl_n_max
        catalog.loc[idx, "wl_k_min"] = wl_k_min
        catalog.loc[idx, "wl_k_max"] = wl_k_max
        catalog.loc[idx, "wl_min"] = wl_min
        catalog.loc[idx, "wl_max"] = wl_max

        df = DataFrame(
            {
                key: val
                for key, val in zip(
                    self._raw_data_columns.keys(), [idx, _cs, _wls_n, _ns, _wls_k, _ks]
                )
            }
        )
        # Arrange the columns according to the order of _raw_data_columns
        df = df.loc[:, self._raw_data_columns.keys()].astype(self._raw_data_columns)
        return df, catalog

    def _create_raw_data_and_modify_catalog(
        self, catalog: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        """Create a DataFrame for experimental data."""
        logger.info("Creating raw data...")
        df = DataFrame(columns=self._raw_data_columns)
        for idx in catalog.index:
            logger.debug("{}: {}".format(idx, catalog.loc[idx, "path"]))
            df_idx, catalog = self._extract_raw_data(idx, catalog)
            df = df.append(df_idx, ignore_index=True)
        df = df.astype(self._raw_data_columns)
        catalog.to_csv(self._catalog_file, index=False, encoding="utf-8")
        df.to_csv(self._raw_data_file, index=False, encoding="utf-8")
        logger.info("Done.")
        return df, catalog

    def load_grid_data(self, id: Optional[int] = None) -> DataFrame:
        """Load grid data of (wl, n, k) for given id.

        Args:
            id (Optional[int]): ID number. If id is None, all the data is loaded.
                Defaults to None.

        Returns:
            DataFrame: Grid data of (wl, n, k).
                (wl, n, k) = (wavelength, refractive index, extinction coefficient).
        """
        if not os.path.isfile(self._grid_data_file):
            logger.warning("Grid data file not found.")
            logger.warning("Creating grid data file...")
            self._create_grid_data(self.catalog, self.raw_data)
            logger.warning("Done.")
        else:
            logger.info("Grid data file found at {}".format(self._grid_data_file))
        if id is None:
            return pd.read_hdf(self._grid_data_file).set_index("id")
        return pd.read_hdf(self._grid_data_file, where=f"id == {id}").set_index("id")

    def _create_grid_data(self, catalog: DataFrame, raw_data: DataFrame) -> None:
        """Create a DataFrame for the wl-nk data."""
        logger.info("Creating grid data...")
        columns = self._grid_data_columns.keys()
        df = DataFrame(columns=columns)
        logger.setLevel(ERROR)
        for idx in catalog.index:
            material = riip.material.RiiMaterial(idx, catalog, raw_data)
            wl_min = material.catalog.loc["wl_min"]
            wl_max = material.catalog.loc["wl_max"]
            wls = np.linspace(wl_min, wl_max, 200)
            ns = material.n(wls)
            ks = material.k(wls)
            data = {key: val for key, val in zip(columns, [idx, wls, ns, ks])}
            df = df.append(DataFrame(data).loc[:, columns], ignore_index=True)
        logger.setLevel(WARNING)
        df = df.astype(self._grid_data_columns)
        df.to_hdf(
            self._grid_data_file,
            "grid_data",
            mode="w",
            data_columns=["id"],
            format="table",
        )
        logger.info("Done.")

    def update_db(self) -> None:
        """Pull repository and update local database."""
        if not os.path.isfile(os.path.join(self._db_path, "library.yml")):
            logger.warning("Cloning Repository.")
            git.Repo.clone_from(_ri_database_repo, self._ri_database, branch="master")
            logger.warning("Done.")
        else:
            logger.warning("Pulling Repository...")
            repo = git.Repo(self._ri_database)
            repo.remotes.origin.pull()
            logger.warning("Done.")
        logger.warning("Updating catalog file...")
        catalog = self._add_my_db_to_catalog(self._create_catalog())
        logger.warning("Done.")
        logger.warning("Updating raw data file...")
        self.raw_data, self.catalog = self._create_raw_data_and_modify_catalog(catalog)
        self.catalog = self.catalog.set_index("id")
        self.raw_data = self.raw_data.set_index("id")
        logger.warning("Done.")
        logger.warning("Updating grid data file...")
        self._create_grid_data(self.catalog, self.raw_data)
        logger.warning("Done.")
        logger.warning("All Done.")

        """."""

    def search(self, name: str) -> DataFrame:
        """Search pages which contain the name of material.

        Args:
            name (str): Name of material

        Returns:
            DataFrame: Simplified catalog
        """
        columns = [
            "book",
            "section",
            "page",
            "formula",
            "tabulated",
            "wl_min",
            "wl_max",
        ]
        df = self.catalog[
            (
                (self.catalog["book"].str.contains(name))
                | (
                    self.catalog["book_name"]
                    .str.replace("<sub>", "")
                    .str.replace("</sub>", "")
                    .str.lower()
                    .str.contains(name.lower())
                )
            )
        ]
        return df.loc[:, columns]

    def select(self, cond: str) -> DataFrame:
        """Select pages that fullfil the condition.

        Args:
            cond (str): Query condition, such as '1.5 <= n <= 2 & 1.0 <= wl <= 2.0'

        Returns:
            List[int]: Simplified catalog
        """
        columns = [
            "book",
            "section",
            "page",
            "formula",
            "tabulated",
            "wl_min",
            "wl_max",
        ]
        gd = self.load_grid_data()
        id_list = gd.query(cond).index.unique()
        return self.catalog.loc[id_list, columns]

    def show(self, id: int | Sequence[int]) -> DataFrame:
        """Summary of page(s) of ID (list of IDs).

        Args:
            id (Union[int, Sequence[int]]): ID number

        Returns:
            DataFrame: Simplified catalog
        """
        columns = [
            "book",
            "section",
            "page",
            "formula",
            "tabulated",
            "wl_min",
            "wl_max",
        ]
        return self.catalog.loc[id, columns]

    def material(self, params: dict) -> riip.material.Material:
        """Create instance of Material class associated with ID.

        Args:
            params (dict): Parameter dict that can contain the following values:
                'id': ID number (int)
                'book': book value in catalog of RiiDataFrame. (str)
                'page': page value in catalog of RiiDataFrame. (str)
                'RI': Constant refractive index. (complex)
                'e': Constant permittivity. (complex)
                'bound_check': True if bound check should be done. Defaults to True. (bool)
                'im_factor': A magnification factor multiplied to the imaginary part of permittivity. Defaults to 1.0. (float)

        Returns:
            Material: A class that provides refractive index, extinction coefficient and dielectric function of the material
        """
        return riip.material.Material(params, self)

    def read(self, id: int, as_dict: bool = False):
        """Return contants of a page associated with the id.

        Args:
            id (int): ID number
            as_dict (bool): If True, the page contents are returned as python dict
        Returns:
            Union[str, dict]: Page contents
        """
        path = self.catalog.loc[id, "path"]
        with open(path) as fd:
            if as_dict:
                contents = yaml.safe_load(fd)
            else:
                contents = fd.read()
        return contents

    def references(self, id: int) -> HTML:
        """Return REFERENCES as IPython.display.HTML class.

        Args:
            id (int): ID number

        Returns:
            HTML: REFERENCES as IPython.display.HTML class
        """
        contents = self.read(id, as_dict=True)
        return HTML(contents["REFERENCES"])

    def plot(
        self,
        id: int,
        comp: str = "n",
        fmt1: Optional[str] = "-",
        fmt2: Optional[str] = "--",
        **kwargs,
    ):
        """Plot refractive index, extinction coefficient or permittivity.

        Args:
            id (int): ID number.
            comp (str): 'n', 'k' or 'eps'.
            fmt1 (Union[str, None]): Plot format for n and Re(eps).
            fmt2 (Union[str, None]): Plot format for k and Im(eps).
        """
        label = self.catalog.loc[id, "page"]
        df = self.load_grid_data(id)
        wls = df["wl"]
        ns = df["n"]
        ks = df["k"]
        kwargs.setdefault("lw", 4)
        kwargs.setdefault("ms", 8)
        if comp == "n":
            plt.plot(wls, ns, fmt1, label=label, **kwargs)
            plt.ylabel(r"$n$")
        elif comp == "k":
            plt.plot(wls, ks, fmt2, label=label, **kwargs)
            plt.ylabel(r"$k$")
        elif comp == "eps":
            eps_r = ns ** 2 - ks ** 2
            eps_i = 2 * ns * ks
            (line,) = plt.plot(wls, eps_r, fmt1, label=label, **kwargs)
            color = line.get_color()
            plt.plot(wls, eps_i, fmt2, color=color, **kwargs)
            plt.ylabel(r"$\varepsilon$")
        plt.xlabel(r"$\lambda$ $[\mathrm{\mu m}]$")
        plt.legend()


if __name__ == "__main__":
    from logging import DEBUG, Formatter, StreamHandler, getLogger

    logger = getLogger("")
    formatter = Formatter(fmt="%(levelname)s:[%(name)s.%(funcName)s]: %(message)s")
    logger.setLevel(DEBUG)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(DEBUG)
    logger.addHandler(stream_handler)

    rii_df = RiiDataFrame()
    rii_df.update_db()
