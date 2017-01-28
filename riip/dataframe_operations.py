#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logging import getLogger
import os
from math import nan
from typing import NewType, Iterable, Any
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
_ri_database_repo = ("https://github.com/mnishida/" +
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

        # Preparing catalog
        if not os.path.isfile(self.catalog_file):
            logger.warning("Catalog file not found.")
            if not os.path.isfile(os.path.join(self.db_path, 'library.yml')):
                logger.warning("Cloning Repository.")
                git.Repo.clone_from(_ri_database_repo, self._ri_database,
                                    branch='master')
                logger.warning("Cloned.")
            self.create_catalog()
            logger.warning("Catalog file has been created.")
        else:
            logger.info("Catalog file found at {}".format(self.catalog_file))
        self.catalog = csv_to_df(self.catalog_file)

        # Preparing raw_data
        if not os.path.isfile(self.raw_data_file):
            logger.warning("Raw-data file not found.")
            self.create_raw_data()
            logger.warning("Raw-data file has been created.")
        else:
            logger.info(
                "Raw-data file found at {}".format(self.raw_data_file))
        self.catalog = csv_to_df(self.catalog_file)
        self.raw_data = csv_to_df(self.raw_data_file)

        # Preparing grid_data
        if not os.path.isfile(self.grid_data_file):
            logger.warning("Grid-data file not found.")
            self.create_grid_data()
            logger.warning("Grid-data file has been created.")
        else:
            logger.info(
                "Grid-data file found at {}".format(self.grid_data_file))
        self.grid_data = csv_to_df(self.grid_data_file)

    def extract_entry(self) -> Iterable[Any]:
        """Yield a single data set."""

        reference_path = os.path.normpath(self.db_path)
        library_file = os.path.join(reference_path, "library.yml")
        with open(library_file, "r", encoding='utf-8') as f:
            logger.debug(library_file)
            catalog = yaml.safe_load(f)
            logger.debug("loaded.")
        idx = 0
        for sh in catalog:
            shelf = sh['SHELF']
            if shelf == '3d':  # This shelf does not seem to contain new data.
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
                            # This DIVIDER specifies the phase of the material
                            # such as gas, liquid or solid, so it is added to
                            # the book and book_name with parentheses.
                            page_class = " ({})".format(p['DIVIDER'])
                        else:
                            book = ''.join([b['BOOK'], page_class])
                            book_name = ''.join([b['name'], page_class])
                            page = p['PAGE']
                            path = os.path.normpath(p['path'])
                            logger.info("{0} {1} {2}".format(idx, book, page))
                            yield [idx, shelf, shelf_name, division, book,
                                   book_name, page, path, '', '', '', '', '',
                                   '']
                            idx += 1

    def create_catalog(self) -> None:
        """Create catalog DataFrame from library.yml."""
        logger.info("Creating catalog...")
        columns = ['id', 'shelf', 'shelf_name', 'division', 'book', 'book_name',
                   'page', 'path', 'formula', 'tabulated', 'wl_min', 'wl_max',
                   'wl_f_min', 'wl_f_max']
        df = pd.DataFrame(self.extract_entry(), columns=columns)
        df.set_index('id', inplace=True, drop=True)
        df.to_csv(self.catalog_file)
        logger.info("Done.")

    def extract_raw_data(self, idx: int) -> PandasDataFrame:
        """Yield a single raw data set.

        Args:
            idx: The ID number of the data set.
        """
        path = os.path.join(self.db_path, self.catalog.loc[idx, 'path'])
        with open(path, "r", encoding='utf-8') as f:
            data_list = yaml.safe_load(f)['DATA']
        wl_f_min = wl_n_min = wl_k_min = 0
        wl_f_max = wl_n_max = wl_k_max = np.inf
        formula = nan
        tabulated = ''
        cs = wls_n = wls_k = ns = ks = np.array([nan])
        for data in data_list:
            data_type, data_set = data['type'].strip().split()
            if data_type == 'tabulated':
                if data_set == 'nk':
                    tabulated += data_set
                    wls_n, ns, ks = np.array(
                        [line.strip().split() for line
                         in data['data'].strip().split('\n')], dtype=float).T
                    inds = np.argsort(wls_n)
                    wls_n = wls_n[inds]
                    wls_k = wls_n
                    ns = ns[inds]
                    ks = ks[inds]
                    wl_n_min = wl_k_min = wls_n[0]
                    wl_n_max = wl_k_max = wls_n[-1]
                elif data_set == 'n':
                    tabulated += data_set
                    wls_n, ns = np.array(
                        [line.strip().split() for line
                         in data['data'].strip().split('\n')], dtype=float).T
                    inds = np.argsort(wls_n)
                    wls_n = wls_n[inds]
                    ns = ns[inds]
                    wl_n_min = wls_n[0]
                    wl_n_max = wls_n[-1]
                elif data_set == 'k':
                    tabulated += data_set
                    wls_k, ks = np.array(
                        [line.strip().split() for line
                         in data['data'].strip().split('\n')], dtype=float).T
                    inds = np.argsort(wls_k)
                    wls_k = wls_k[inds]
                    ks = ks[inds]
                    wl_k_min = wls_k[0]
                    wl_k_max = wls_k[-1]
                else:
                    raise Exception("DATA is broken.")
            elif data_type == 'formula':
                formula = data_set
                wl_f_min, wl_f_max = [float(s) for s in
                                      data['range'].strip().split()]
                cs = np.array(data['coefficients'].strip().split(), dtype=float)
            else:
                raise Exception(
                    "DATA has unknown contents {}".format(data_type))
        if len(tabulated) > 2:
            raise Exception("Too much tabulated data are provided")
        elif len(tabulated) == 2:
            if 'n' not in tabulated or 'k' not in tabulated:
                raise Exception("There is redundancy in n or k.")
            elif tabulated == 'kn':
                tabulated = 'nk'
        num_n = len(ns)
        num_k = len(ks)
        wl_min = max(wl_f_min, wl_n_min, wl_k_min)
        wl_max = min(wl_f_max, wl_n_max, wl_k_max)
        if wl_f_min == 0:
            wl_f_min = wl_f_max = nan
            cs = [nan]
            num_c = 1
        else:
            num_c = len(cs)
            cs = np.concatenate([cs, np.array([0.0] * (17 - num_c))])
            num_c = 17
        num = max(num_n, num_k, num_c)
        cs = np.concatenate([cs, np.array([nan] * (num - num_c))])
        wls_n = np.concatenate(
            [wls_n, np.array([nan] * (num - num_n))])
        ns = np.concatenate([ns, np.array([nan] * (num - num_n))])
        wls_k = np.concatenate(
            [wls_k, np.array([nan] * (num - num_k))])
        ks = np.concatenate([ks, np.array([nan] * (num - num_k))])
        self.catalog.loc[idx, 'formula'] = formula
        self.catalog.loc[idx, 'tabulated'] = tabulated
        self.catalog.loc[idx, 'wl_f_min'] = wl_f_min
        self.catalog.loc[idx, 'wl_f_max'] = wl_f_max
        self.catalog.loc[idx, 'wl_min'] = wl_min
        self.catalog.loc[idx, 'wl_max'] = wl_max
        df = pd.DataFrame(
            {'id': idx, 'cs': cs,
             'wls_n': wls_n, 'ns': ns, 'wls_k': wls_k, 'ks': ks})
        columns = ['id', 'cs', 'wls_n', 'ns', 'wls_k', 'ks']
        return df.ix[:, columns]

    def create_raw_data(self) -> None:
        """Create a DataFrame for experimental data."""
        logger.info("Creating raw data...")
        columns = ['id', 'cs', 'wls_n', 'ns', 'wls_k', 'ks']
        df = pd.DataFrame(columns=columns)
        for idx in self.catalog.index:
            logger.info("{}: {}".format(idx, self.catalog.loc[idx, 'path']))
            df = df.append(self.extract_raw_data(idx),
                           ignore_index=True)
        self.catalog.set_index('id', inplace=True, drop=True)
        self.catalog.to_csv(self.catalog_file)
        df['id'] = df['id'].astype(int)
        df.set_index('id', inplace=True, drop=True)
        df.to_csv(self.raw_data_file)
        logger.info("Done.")

    def create_grid_data(self) -> None:
        """Create a DataFrame for the wl-nk data."""
        logger.info("Creating grid data...")
        columns = ['id', 'wl', 'n', 'k']
        df = pd.DataFrame(columns=columns)
        for idx in set(self.raw_data['id']):
            catalog = self.catalog.loc[idx]
            data = self.raw_data[self.raw_data['id'] == idx]
            wl_min = catalog.loc['wl_min']
            wl_max = catalog.loc['wl_max']
            wls = np.linspace(wl_min, wl_max, 129)
            func = DispersionFormula(catalog, data)
            ns, ks = func(wls)
            df = df.append(pd.DataFrame(
                {'id': idx, 'wl': wls, 'n': ns, 'k': ks}).ix[:, columns],
                           ignore_index=True)
        df['id'] = df['id'].astype(int)
        df.set_index('id', inplace=True, drop=True)
        df.to_csv(self.grid_data_file)
        logger.info("Done.")


def csv_to_df(csv_file: str) -> PandasDataFrame:
    """Convert csv file to a DataFrame."""
    logger.info("Loading {}".format(os.path.basename(csv_file)))
    df = pd.read_csv(csv_file)
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
    print(rii_df.catalog.loc[0])
    print(rii_df.catalog.index)
    print(rii_df.raw_data.head)
