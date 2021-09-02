#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from logging import getLogger
from typing import Any, ClassVar, Dict, Iterable, List, Sequence, Tuple, Union

import git
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame

from riip.material import Material

logger = getLogger(__package__)
_dirname = os.path.dirname(__file__)
_ri_database = os.path.join(_dirname, "data", "refractiveindex.info-database")
_db_directory = os.path.join(_ri_database, "database")
_my_db_directory = os.path.join(_dirname, "data", "my_database")
_catalog_file = os.path.join(_dirname, "data", "catalog.csv")
_raw_data_file = os.path.join(_dirname, "data", "raw_data.csv")
_grid_data_file = os.path.join(_dirname, "data", "grid_data.csv")
_ri_database_repo = (
    "https://github.com/polyanskiy/" + "refractiveindex.info-database.git"
)
_ri_database_patch = os.path.join(_dirname, "..", "riid.patch")


class RiiDataFrame:
    """This class provides a Pandas DataFrame for 'refractiveindex.info database'.

    Attributes:
        db_path: The path to the refractiveindex.info-database/database.
        my_db_path: The path to my_database.
        catalog: The catalog.
        catalog_file: The csv filename to store the catalog.
        raw_data: The experimental data.
        raw_data_file: The csv filename to store the raw_data.
        grid_data_file: The csv filename to store the grid_data.
    """

    _catalog_columns: ClassVar[OrderedDict] = OrderedDict(
        (
            ("id", int),
            ("shelf", str),
            ("shelf_name", str),
            ("division", str),
            ("book", str),
            ("book_name", str),
            ("page", str),
            ("path", str),
            ("formula", int),
            ("tabulated", str),
            ("num_n", int),
            ("num_k", int),
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
            ("id", int),
            ("c", np.float64),
            ("wl_n", np.float64),
            ("n", np.float64),
            ("wl_k", np.float64),
            ("k", np.float64),
        )
    )

    _grid_data_columns: ClassVar[OrderedDict] = OrderedDict(
        (("id", int), ("wl", np.float64), ("n", np.float64), ("k", np.float64))
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
        self.db_path: str = db_path
        self.my_db_path: str = my_db_path
        self._ri_database: str = os.path.dirname(self.db_path)
        self.catalog_file: str = catalog_file
        self.raw_data_file: str = raw_data_file
        self.grid_data_file: str = grid_data_file
        _catalog, _raw_data = self.load_catalog_and_raw_data()
        self.catalog: DataFrame = _catalog
        self.raw_data: DataFrame = _raw_data

    def load_catalog_and_raw_data(self) -> Tuple[DataFrame, DataFrame]:
        # Create csv files
        if not os.path.isfile(self.catalog_file):
            logger.warning("Catalog file not found.")
            if not os.path.isfile(os.path.join(self.db_path, "library.yml")):
                logger.warning("Cloning Repository...")
                # repo = git.Repo.clone_from(
                #     _ri_database_repo, self._ri_database, branch="master"
                # )
                # repo.git.apply(_ri_database_patch)
                git.Repo.clone_from(
                    _ri_database_repo, self._ri_database, branch="master"
                )
                logger.warning("Done.")
            logger.warning("Creating catalog file...")
            catalog = self.add_my_db_to_catalog(self.create_catalog())
            logger.warning("Done.")

            # Preparing raw_data
            logger.warning("Creating raw data file...")
            raw_data, catalog = self.create_raw_data_and_modify_catalog(catalog)
            logger.warning("Done.")

            # Preparing grid_data
            logger.warning("Updating grid data file...")
            self.create_grid_data(catalog, raw_data)
            logger.warning("Done.")
        catalog = load_csv(self.catalog_file, dtype=self._catalog_columns)
        raw_data = load_csv(self.raw_data_file, dtype=self._raw_data_columns)
        return catalog, raw_data

    @staticmethod
    def extract_entry(db_path: str, start_id: int = 0) -> Iterable[Any]:
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
                            page_class = ""
                        for p in b["content"]:
                            if "DIVIDER" in p:
                                # This DIVIDER specifies the phase of the
                                #  material such as gas, liquid or solid, so it
                                #  is added to the book and book_name with
                                #  parentheses.
                                page_class = " ({})".format(p["DIVIDER"])
                            else:
                                book = "".join([b["BOOK"], page_class])
                                book_name = "".join([b["name"], page_class])
                                page = p["PAGE"]
                                path = os.path.join(
                                    reference_path, "data", os.path.normpath(p["data"])
                                )
                                logger.debug("{0} {1} {2}".format(idx, book, page))
                                yield [
                                    idx,
                                    shelf,
                                    shelf_name,
                                    division,
                                    book,
                                    book_name,
                                    page,
                                    path,
                                    0,
                                    "f",
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ]
                                idx += 1
        except Exception as e:
            message = (
                "There seems to be some inconsistency in the library.yml "
                + "around id={}, shelf={}, book={}, page={}.".format(
                    idx, shelf, book, page
                )
            )
            raise Exception(message) from e

    def create_catalog(self) -> DataFrame:
        """Create catalog DataFrame from library.yml."""
        logger.info("Creating catalog...")
        df = DataFrame(
            self.extract_entry(self.db_path), columns=self._catalog_columns.keys()
        )
        set_columns_dtype(df, self._catalog_columns)
        logger.info("Done.")
        return df

    def add_my_db_to_catalog(self, catalog) -> DataFrame:
        """Add data in my_database to catalog DataFrame."""
        logger.info("Adding my_db to catalog...")
        start_id = catalog["id"].values[-1] + 1
        logger.debug(start_id)
        df = DataFrame(
            self.extract_entry(self.my_db_path, start_id),
            columns=self._catalog_columns.keys(),
        )
        set_columns_dtype(df, self._catalog_columns)
        df = catalog.append(df, ignore_index=True)
        logger.info("Done.")
        return df

    def load_raw_data(self) -> DataFrame:
        # Load catalog and experimental data.
        df = load_csv(self.raw_data_file, dtype=self._raw_data_columns)
        return df

    def extract_raw_data(
        self, idx: int, catalog: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Yield a single raw data set.

        Some data are inserted into the catalog.
        Args:
            catalog: The catalog.
            idx: The ID number of the data set.
        """
        path = catalog.loc[idx, "path"]
        with open(path, "r", encoding="utf-8") as f:
            data_list = yaml.safe_load(f)["DATA"]
        wl_n_min = wl_k_min = 0
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
        cs = np.array(cs + [0.0] * (num - num_c), dtype=np.float64)
        wls_n = np.array(wls_n + [0.0] * (num - num_n), dtype=np.float64)
        ns = np.array(ns + [0.0] * (num - num_n), dtype=np.float64)
        wls_k = np.array(wls_k + [0.0] * (num - num_k), dtype=np.float64)
        ks = np.array(ks + [0.0] * (num - num_k), dtype=np.float64)

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
                    self._raw_data_columns.keys(), [idx, cs, wls_n, ns, wls_k, ks]
                )
            }
        )
        # Arrange the columns according to the order of _raw_data_columns
        df = df.loc[:, self._raw_data_columns.keys()]
        set_columns_dtype(df, self._raw_data_columns)
        return df, catalog

    def create_raw_data_and_modify_catalog(
        self, catalog: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Create a DataFrame for experimental data."""
        logger.info("Creating raw data...")
        df = DataFrame(columns=self._raw_data_columns)
        for idx in catalog.index:
            logger.debug("{}: {}".format(idx, catalog.loc[idx, "path"]))
            df_idx, catalog = self.extract_raw_data(idx, catalog)
            df = df.append(df_idx, ignore_index=True)
        set_columns_dtype(df, self._raw_data_columns)
        catalog.to_csv(self.catalog_file, index=False, encoding="utf-8")
        df.to_csv(self.raw_data_file, index=False, encoding="utf-8")
        logger.info("Done.")
        return df, catalog

    def load_grid_data(self) -> DataFrame:
        if not os.path.isfile(self.grid_data_file):
            logger.warning("Grid data file not found.")
            logger.warning("Creating grid data file...")
            self.create_grid_data(self.catalog, self.raw_data)
            logger.warning("Done.")
        else:
            logger.info("Grid data file found at {}".format(self.grid_data_file))
        return load_csv(self.grid_data_file, dtype=self._grid_data_columns)

    def create_grid_data(self, catalog: DataFrame, raw_data: DataFrame) -> None:
        """Create a DataFrame for the wl-nk data."""
        logger.info("Creating grid data...")
        columns = self._grid_data_columns.keys()
        df = DataFrame(columns=columns)
        for idx in set(raw_data["id"]):
            a_catalog = catalog.loc[idx]
            data = raw_data[raw_data["id"] == idx]
            material = Material(a_catalog, data)
            wl_min = a_catalog.loc["wl_min"]
            wl_max = a_catalog.loc["wl_max"]
            wls = np.linspace(wl_min, wl_max, 200)
            ns = material.n(wls)
            ks = material.k(wls)
            data = {key: val for key, val in zip(columns, [idx, wls, ns, ks])}
            df = df.append(DataFrame(data).loc[:, columns], ignore_index=True)
        set_columns_dtype(df, self._grid_data_columns)
        df.to_csv(self.grid_data_file, index=False, encoding="utf-8")
        logger.info("Done.")

    def update_db(self) -> None:
        if not os.path.isfile(os.path.join(self.db_path, "library.yml")):
            logger.warning("Cloning Repository.")
            git.Repo.clone_from(_ri_database_repo, self._ri_database, branch="master")
            logger.warning("Done.")
        else:
            logger.warning("Pulling Repository...")
            repo = git.Repo(self._ri_database)
            repo.remotes.origin.pull()
            logger.warning("Done.")
        logger.warning("Updating catalog file...")
        catalog = self.add_my_db_to_catalog(self.create_catalog())
        logger.warning("Done.")
        logger.warning("Updating raw data file...")
        self.raw_data, self.catalog = self.create_raw_data_and_modify_catalog(catalog)
        logger.warning("Done.")
        logger.warning("Updating grid data file...")
        self.create_grid_data(self.catalog, self.raw_data)
        logger.warning("Done.")
        logger.warning("All Done.")

    def search(self, name: str) -> DataFrame:
        """Search pages which contain the name."""
        columns = [
            "shelf",
            "book",
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

    def select(self, cond: str) -> List[int]:
        """Select pages which fulfill the condition."""
        columns = [
            "shelf",
            "book",
            "page",
            "formula",
            "tabulated",
            "wl_min",
            "wl_max",
        ]
        gd = self.load_grid_data()
        id_list = gd.query(cond).index.unique()
        return self.catalog.loc[id_list, columns]

    def show(self, id: Union[int, Sequence[int]]) -> DataFrame:
        """Show page(s) of the ID (list of IDs)."""
        columns = ["shelf", "book", "page", "formula", "tabulated", "wl_min", "wl_max"]
        return self.catalog.loc[id, columns]

    def material(self, id: int, bound_check: bool = True) -> Material:
        """Material associated with the ID."""
        return Material(self.catalog.loc[id], self.raw_data.loc[id], bound_check)


def load_csv(csv_file: str, dtype: Union[None, Dict] = None) -> DataFrame:
    """Convert csv file to a DataFrame."""
    # logger.info("Loading {}".format(os.path.basename(csv_file)))
    df = pd.read_csv(csv_file, dtype=dtype, index_col="id")
    # logger.info("Done.")
    return df


def set_columns_dtype(df: DataFrame, columns: Dict):
    """Set data type of each column in the DataFrame."""
    for key, val in columns.items():
        df[key] = df[key].astype(val)


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
    # print(rii_df.catalog.loc[0])
    # print(rii_df.catalog.index)
    # print(rii_df.raw_data.head)

    rii_df.update_db()
    # catalog = rii_df.add_my_db_to_catalog(
    #     rii_df.create_catalog())
    # raw_data, catalog = rii_df.create_raw_data_and_modify_catalog(catalog)
    # rii_df.catalog = csv_to_df(rii_df.catalog_file)
    # rii_df.create_raw_data()
    # rii_df.raw_data = csv_to_df(rii_df.raw_data_file)
    # rii_df.create_grid_data()
