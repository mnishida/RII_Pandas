#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logging import getLogger
import os
from math import nan
from collections import OrderedDict
from typing import NewType, Iterable, Any, Union, Dict
import git
import yaml
import pandas as pd
import numpy as np
from riip.dispersion_formulas import DispersionFormula

__all__ = ['RiiDataFrame', 'csv_to_df']
logger = getLogger(__package__)
PandasDataFrame = NewType('PandasDataFrame', pd.DataFrame)

_dirname = os.path.dirname(os.path.dirname(__file__))
_ri_database = os.path.join(
    _dirname, 'data', 'refractiveindex.info-database')
_db_directory = os.path.join(_ri_database, 'database')
_catalog_file = os.path.join(_dirname, 'data', 'catalog.csv')
_raw_data_file = os.path.join(_dirname, 'data', 'raw_data.csv')
_grid_data_file = os.path.join(_dirname, 'data', 'grid_data.csv')
_ri_database_repo = ("https://github.com/polyanskiy/" +
                     "refractiveindex.info-database.git")


class RiiDataFrame:
    """This class provides a Pandas DataFrame for 'refractiveindex.info database'.

    Attributes:
        db_path (str): The path to the refractiveindex.info-database/database.
        catalog (DataFrame): The catalog.
        catalog_file (str): The csv filename to store the catalog.
        raw_data (DataFrame): The experimental data.
        raw_data_file (str): The csv filename to store the raw_data.
        grid_data (DataFrame): The grid wl-nk data.
        grid_data_file (str): The csv filename to store the grid_data.
    """

    def __init__(self, db_path: str = _db_directory,
                 catalog_file: str = _catalog_file,
                 raw_data_file: str = _raw_data_file,
                 grid_data_file: str = _grid_data_file):
        """Initialize the RiiDataFrame.

        Args:
            db_path: The path to the refractiveindex.info-database/database.
            catalog_file: The filename of the catalog csv file.
            raw_data_file: The filename of the experimental data csv file.
            grid_data_file: The filename of the grid wl-nk data csv file.
        """
        self.db_path = db_path
        self._ri_database = os.path.dirname(self.db_path)
        self.catalog_file = catalog_file
        self.raw_data_file = raw_data_file
        self.grid_data_file = grid_data_file
        self._catalog_columns = OrderedDict((
            ('id', int), ('shelf', str), ('shelf_name', str), ('division', str),
            ('book', str), ('book_name', str), ('page', str), ('path', str),
            ('formula', str), ('tabulated', str), ('num_n', int),
            ('num_k', int), ('wl_n_min', np.float64), ('wl_n_max', np.float64),
            ('wl_k_min', np.float64), ('wl_k_max', np.float64)))
        self._raw_data_columns = OrderedDict((
            ('id', int), ('c', np.float64), ('wl_n', np.float64),
            ('n', np.float64), ('wl_k', np.float64), ('k', np.float64)))
        self._grid_data_columns = OrderedDict((
            ('id', int), ('wl', np.float64),
            ('n', np.float64), ('k', np.float64)))

        # Preparing catalog
        if not os.path.isfile(self.catalog_file):
            logger.warning("Catalog file not found.")
            if not os.path.isfile(os.path.join(self.db_path, 'library.yml')):
                logger.warning("Cloning Repository...")
                git.Repo.clone_from(_ri_database_repo, self._ri_database,
                                    branch='master')
                logger.warning("Done.")
            logger.warning("Creating catalog file...")
            self.create_catalog()
            logger.warning("Done.")
        else:
            logger.info("Catalog file found at {}".format(self.catalog_file))
        self.catalog = csv_to_df(self.catalog_file, dtype=self._catalog_columns)

        # Preparing raw_data
        if not os.path.isfile(self.raw_data_file):
            logger.warning("Raw data file not found.")
            logger.warning("Creating raw data file...")
            self.create_raw_data()
            self.catalog = csv_to_df(
                self.catalog_file, dtype=self._catalog_columns)
            logger.warning("Done.")
        else:
            logger.info(
                "Raw data file found at {}".format(self.raw_data_file))
        self.raw_data = csv_to_df(
            self.raw_data_file, dtype=self._raw_data_columns)

        # Preparing grid_data
        if not os.path.isfile(self.grid_data_file):
            logger.warning("Grid data file not found.")
            logger.warning("Creating grid data file...")
            self.create_grid_data()
            logger.warning("Done.")
        else:
            logger.info(
                "Grid data file found at {}".format(self.grid_data_file))
        self.grid_data = csv_to_df(
            self.grid_data_file, dtype=self._grid_data_columns)

    def extract_entry(self) -> Iterable[Any]:
        """Yield a single data set."""

        reference_path = os.path.normpath(self.db_path)
        library_file = os.path.join(reference_path, "library.yml")
        with open(library_file, "r", encoding='utf-8') as f:
            catalog = yaml.safe_load(f)
        idx = 0
        shelf = 'main'
        book = 'Ag (Experimental data)'
        page = 'Johnson'
        try:
            for sh in catalog:
                shelf = sh['SHELF']
                if shelf == '3d':
                    # This shelf does not seem to contain new data.
                    break
                shelf_name = sh['name']
                division = None
                for b in sh['content']:
                    if 'DIVIDER' in b:
                        division = b['DIVIDER']
                    else:
                        if division is None:
                            raise Exception(
                                "'DIVIDER' is missing in 'library.yml'.")
                        if 'DIVIDER' not in b['content']:
                            page_class = ''
                        for p in b['content']:
                            if 'DIVIDER' in p:
                                # This DIVIDER specifies the phase of the
                                #  material such as gas, liquid or solid, so it
                                #  is added to the book and book_name with
                                #  parentheses.
                                page_class = " ({})".format(p['DIVIDER'])
                            else:
                                book = ''.join([b['BOOK'], page_class])
                                book_name = ''.join([b['name'], page_class])
                                page = p['PAGE']
                                path = os.path.normpath(p['path'])
                                logger.debug("{0} {1} {2}".format(
                                    idx, book, page))
                                yield [idx, shelf, shelf_name, division, book,
                                       book_name, page, path,
                                       0, '', 0, 0, '', '', '', '']
                                idx += 1
        except Exception as e:
            message = (
                "There seems to be some inconsistency in the library.yml " +
                "around id={}, shelf={}, book={}, page={}.".format(
                    idx, shelf, book, page))
            raise Exception(message) from e

    def create_catalog(self) -> None:
        """Create catalog DataFrame from library.yml."""
        logger.info("Creating catalog...")
        df = pd.DataFrame(self.extract_entry(),
                          columns=self._catalog_columns.keys())
        df.to_csv(self.catalog_file)
        logger.info("Done.")

    def extract_raw_data(self, idx: int) -> PandasDataFrame:
        """Yield a single raw data set.

        Some data are inserted into the catalog.
        Args:
            idx: The ID number of the data set.
        """
        path = os.path.join(self.db_path, self.catalog.loc[idx, 'path'])
        with open(path, "r", encoding='utf-8') as f:
            data_list = yaml.safe_load(f)['DATA']
        wl_n_min = wl_k_min = 0
        wl_n_max = wl_k_max = np.inf
        formula = ''
        tabulated = ''
        cs = []
        wls_n = []
        wls_k = []
        ns = []
        ks = []
        num_n = num_k = 0
        for data in data_list:
            data_type, data_set = data['type'].strip().split()

            # For tabulated data
            if data_type == 'tabulated':
                if data_set == 'nk':
                    tabulated += data_set
                    wls_n, ns, ks = np.array(
                        [line.strip().split() for line
                         in data['data'].strip().split('\n')], dtype=float).T
                    inds = np.argsort(wls_n)
                    wls_n = list(wls_n[inds])
                    wls_k = wls_n
                    ns = list(ns[inds])
                    ks = list(ks[inds])
                    wl_n_min = wl_k_min = wls_n[0]
                    wl_n_max = wl_k_max = wls_n[-1]
                    num_n = len(wls_n)
                    num_k = len(wls_k)
                elif data_set == 'n':
                    tabulated += data_set
                    wls_n, ns = np.array(
                        [line.strip().split() for line
                         in data['data'].strip().split('\n')], dtype=float).T
                    inds = np.argsort(wls_n)
                    wls_n = list(wls_n[inds])
                    ns = list(ns[inds])
                    wl_n_min = wls_n[0]
                    wl_n_max = wls_n[-1]
                    num_n = len(wls_n)
                elif data_set == 'k':
                    tabulated += data_set
                    wls_k, ks = np.array(
                        [line.strip().split() for line
                         in data['data'].strip().split('\n')], dtype=float).T
                    inds = np.argsort(wls_k)
                    wls_k = list(wls_k[inds])
                    ks = list(ks[inds])
                    wl_k_min = wls_k[0]
                    wl_k_max = wls_k[-1]
                    num_k = len(wls_k)
                else:
                    raise Exception("DATA is broken.")

            # For formulas
            elif data_type == 'formula':
                formula = data_set
                wl_n_min, wl_n_max = [float(s) for s in
                                      data['range'].strip().split()]
                cs = [float(s) for s in data['coefficients'].strip().split()]
            else:
                raise Exception(
                    "DATA has unknown contents {}".format(data_type))

        if len(tabulated) > 2:
            raise Exception("Too many tabulated data set are provided")
        elif 'nn' in tabulated or 'kk' in tabulated:
            raise Exception("There is redundancy in n or k.")
        elif tabulated == 'kn':
            tabulated = 'nk'

        # The coefficients not included in the formula must be zero.
        num_c = len(cs)
        if formula != '':
            cs += [0.0] * (17 - num_c)
            num_c = 17

        # All the arrays must have the same length.
        num = max(num_n, num_k, num_c)
        cs = np.array(cs + [nan] * (num - num_c), dtype=np.float64)
        wls_n = np.array(wls_n + [nan] * (num - num_n), dtype=np.float64)
        ns = np.array(ns + [nan] * (num - num_n), dtype=np.float64)
        wls_k = np.array(wls_k + [nan] * (num - num_k), dtype=np.float64)
        ks = np.array(ks + [nan] * (num - num_k), dtype=np.float64)

        # Obtained data are inserted to the catalog
        self.catalog.loc[idx, 'formula'] = formula
        self.catalog.loc[idx, 'tabulated'] = tabulated
        self.catalog.loc[idx, 'num_n'] = num_n
        self.catalog.loc[idx, 'num_k'] = num_k
        self.catalog.loc[idx, 'wl_n_min'] = wl_n_min
        self.catalog.loc[idx, 'wl_n_max'] = wl_n_max
        self.catalog.loc[idx, 'wl_k_min'] = wl_k_min
        self.catalog.loc[idx, 'wl_k_max'] = wl_k_max

        df = pd.DataFrame(
            {key: val for key, val in
             zip(self._raw_data_columns.keys(),
                 [idx, cs, wls_n, ns, wls_k, ks])})
        # Arrange the columns according to the the order of _raw_data_columns
        return df.ix[:, self._raw_data_columns]

    def create_raw_data(self) -> None:
        """Create a DataFrame for experimental data."""
        logger.info("Creating raw data...")
        df = pd.DataFrame(columns=self._raw_data_columns)
        for idx in self.catalog.index:
            logger.debug("{}: {}".format(idx, self.catalog.loc[idx, 'path']))
            df = df.append(self.extract_raw_data(idx),
                           ignore_index=True)
        self.catalog['num_n'] = self.catalog['num_n'].astype(int)
        self.catalog['num_k'] = self.catalog['num_k'].astype(int)
        self.catalog.to_csv(self.catalog_file)
        df['id'] = df['id'].astype(int)
        df.to_csv(self.raw_data_file)
        logger.info("Done.")

    def create_grid_data(self) -> None:
        """Create a DataFrame for the wl-nk data."""
        logger.info("Creating grid data...")
        columns = self._grid_data_columns.keys()
        df = pd.DataFrame(columns=columns)
        for idx in set(self.raw_data['id']):
            catalog = self.catalog.loc[idx]
            data = self.raw_data[self.raw_data['id'] == idx]
            dispersion = DispersionFormula(catalog, data)
            wl_min = max(float(catalog.loc['wl_n_min']),
                         float(catalog.loc['wl_k_min']))
            wl_max = min(float(catalog.loc['wl_n_max']),
                         float(catalog.loc['wl_k_max']))
            wls = np.linspace(wl_min, wl_max, 257)
            ns = dispersion.func_n(wls)
            ks = dispersion.func_k(wls)
            data = {key: val for key, val in zip(columns, [idx, wls, ns, ks])}
            df = df.append(pd.DataFrame(data).ix[:, columns], ignore_index=True)
        df['id'] = df['id'].astype(int)
        df.to_csv(self.grid_data_file)
        logger.info("Done.")

    def update_db(self) -> None:
        if not os.path.isfile(os.path.join(self.db_path, 'library.yml')):
            logger.warning("Cloning Repository.")
            git.Repo.clone_from(_ri_database_repo, self._ri_database,
                                branch='master')
            logger.warning("Done.")
        else:
            logger.warning("Pulling Repository...")
            repo = git.Repo(self._ri_database)
            repo.remotes.origin.pull()
            logger.warning("Done.")
        logger.warning("Updating catalog file...")
        self.create_catalog()
        self.catalog = csv_to_df(
            self.catalog_file, dtype=self._catalog_columns)
        logger.warning("Done.")
        logger.warning("Updating raw data file...")
        self.create_raw_data()
        self.raw_data = csv_to_df(
            self.raw_data_file, dtype=self._raw_data_columns)
        logger.warning("Done.")
        logger.warning("Updating grid data file...")
        self.create_grid_data()
        self.grid_data = csv_to_df(
            self.grid_data_file, self._grid_data_columns)
        logger.warning("Done.")
        logger.warning("All Done.")


def csv_to_df(csv_file: str,
              dtype: Union[None, Dict] = None) -> PandasDataFrame:
    """Convert csv file to a DataFrame."""
    logger.info("Loading {}".format(os.path.basename(csv_file)))
    df = pd.read_csv(csv_file, dtype=dtype)
    logger.info("Done.")
    return df


if __name__ == '__main__':
    from logging import getLogger, StreamHandler, Formatter, DEBUG
    logger = getLogger('')
    formatter = Formatter(
        fmt='%(levelname)s:[%(name)s.%(funcName)s]: %(message)s')
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
    # rii_df.create_catalog()
    # rii_df.catalog = csv_to_df(rii_df.catalog_file)
    # rii_df.create_raw_data()
    # rii_df.catalog = csv_to_df(rii_df.catalog_file)
    # rii_df.create_raw_data()
    # rii_df.raw_data = csv_to_df(rii_df.raw_data_file)
    # rii_df.create_grid_data()
