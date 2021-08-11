#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
from filecmp import cmp

import pandas as pd
from numpy.testing import assert_allclose

import riip


class KnownValues(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.db_directory = os.path.join(dirname, "data")
        self.catalog_file_known = os.path.join(dirname, "catalog.csv")
        self.raw_data_file_known = os.path.join(dirname, "raw_data.csv")
        self.grid_data_file_known = os.path.join(dirname, "grid_data.csv")
        self.catalog_file = os.path.join(dirname, "data", "catalog.csv")
        self.raw_data_file = os.path.join(dirname, "data", "raw_data.csv")
        self.grid_data_file = os.path.join(dirname, "data", "grid_data.csv")
        self.my_db_directory = os.path.join(dirname, "data", "my_database")
        self.ri = riip.RiiDataFrame(
            self.db_directory,
            self.catalog_file,
            self.raw_data_file,
            self.grid_data_file,
            self.my_db_directory,
        )

    def test_catalog(self):
        """Check if the catalog is created as expected."""
        dirname = os.path.dirname(__file__)
        catalog = riip.load_csv(self.catalog_file, self.ri._catalog_columns)
        catalog_known = riip.load_csv(self.catalog_file_known, self.ri._catalog_columns)
        for ind in catalog_known.index:
            path = catalog_known.loc[ind, "path"].replace("/", os.sep)
            catalog_known.loc[ind, "path"] = dirname + path
        pd.testing.assert_frame_equal(catalog, catalog_known)

    def test_raw_data(self):
        """Check if the raw data is created as expected."""
        self.assertTrue(cmp(self.raw_data_file, self.raw_data_file_known))

    def test_grid_data(self):
        """Check if the grid data is created as expected."""
        grid_data = riip.load_csv(self.grid_data_file, self.ri._grid_data_columns)
        grid_data_known = riip.load_csv(
            self.grid_data_file_known, self.ri._grid_data_columns
        )
        pd.testing.assert_frame_equal(grid_data, grid_data_known)

    def test_compare_with_RIID(self):
        """Compare withe csv data of Refractiveindex.infor database."""
        catalog = self.ri.catalog
        for idx in catalog.index:
            root, ext = os.path.splitext(catalog.loc[idx, "path"])
            df = pd.read_csv(root + ".csv", header=0)
            material = self.ri.material(idx)
            ns = material.n(df["wl"].values)
            print(catalog.loc[idx, "page"])
            assert_allclose(df["n"].values, ns, rtol=1e-5)

    def tearDown(self):
        os.remove(self.catalog_file)
        os.remove(self.raw_data_file)
        os.remove(self.grid_data_file)


if __name__ == "__main__":
    unittest.main()
