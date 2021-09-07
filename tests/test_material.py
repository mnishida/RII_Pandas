#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def test_material(num_regression):
    from riip import Material

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
