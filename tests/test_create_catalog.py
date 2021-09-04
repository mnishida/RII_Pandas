import unittest

import numpy.testing as npt
import pandas as pd

import riip


def test_create_catalog():
    ri = riip.RiiDataFrame()
    dupl = ri.catalog.duplicated(subset=["book", "page"], keep=False)
    if dupl.sum():
        print("Found duplication:")
        print(ri.catalog.loc[dupl, ["book", "page"]])
        raise Exception

    for _id in ri.catalog.query("'n' in tabulated").index:
        try:
            num_n = ri.catalog.loc[_id, "num_n"]
            wl_n = ri.raw_data.loc[_id, ["wl_n"]].to_numpy()[:num_n]
            n1 = ri.raw_data.loc[_id, ["n"]].to_numpy()[:num_n]
            n2 = ri.material(_id, bound_check=False).n(wl_n)
            npt.assert_array_almost_equal(n1, n2)
        except Exception as e:
            print(_id)
            raise e

    for _id in ri.catalog.query("'k' in tabulated").index:
        try:
            num_k = ri.catalog.loc[_id, "num_k"]
            wl_k = ri.raw_data.loc[_id, ["wl_k"]].to_numpy()[:num_k]
            k1 = ri.raw_data.loc[_id, ["k"]].to_numpy()[:num_k]
            k2 = ri.material(_id, bound_check=False).k(wl_k)
            npt.assert_array_almost_equal(k1, k2)
        except Exception as e:
            print(_id)
            raise e
