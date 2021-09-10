#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt

import riip
from riip import Material


def test_material(num_regression):

    w = 2.0 * np.pi
    data = {
        "air": Material({"RI": 1.0})(w),
        "GaN": Material({"book": "GaN", "page": "Barker-o"})(w),
        "W": Material({"book": "W", "page": "Rakic-DLF"})(w),
        "Au-D": Material({"book": "Au", "page": "Vial-DF"})(w),
        "Au-DL": Material({"book": "Au", "page": "Stewart-DLF"})(w),
        "Ag": Material({"book": "Ag", "page": "Vial-DLF"})(w),
        "Al": Material({"book": "Al", "page": "Rakic-DLF"})(w),
        "Al-lowloss": Material({"book": "Al", "page": "Rakic-DLF", "im_factor": 0.1})(
            w
        ),
    }
    print(data)
    num_regression.check(data)


def test_n_and_k():
    rid = riip.RiiDataFrame()

    for _id in rid.catalog.query("'n' in tabulated").index:
        book = rid.catalog.loc[_id, "book"]
        page = rid.catalog.loc[_id, "page"]
        material = Material({"book": book, "page": page, "bound_check": False}, rid)
        try:
            num_n = rid.catalog.loc[_id, "num_n"]
            wl_n = rid.raw_data.loc[_id, ["wl_n"]].to_numpy()[:num_n]
            n1 = rid.raw_data.loc[_id, ["n"]].to_numpy()[:num_n]
            n2 = material.n(wl_n)
            npt.assert_array_almost_equal(
                abs(n1), n2
            )  # abs is necessary only for metamaterial

        except Exception as e:
            print(_id)
            raise e

    for _id in rid.catalog.query("'k' in tabulated").index:
        book = rid.catalog.loc[_id, "book"]
        page = rid.catalog.loc[_id, "page"]
        material = Material({"book": book, "page": page, "bound_check": False}, rid)
        try:
            num_k = rid.catalog.loc[_id, "num_k"]
            wl_k = rid.raw_data.loc[_id, ["wl_k"]].to_numpy()[:num_k]
            k1 = rid.raw_data.loc[_id, ["k"]].to_numpy()[:num_k]
            k2 = material.k(wl_k)
            npt.assert_array_almost_equal(
                k1 * (k1 > 0), k2
            )  # negative k for HIKARI SK-2. Why?
        except Exception as e:
            print(_id)
            raise e
