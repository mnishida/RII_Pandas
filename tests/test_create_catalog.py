import numpy as np
import numpy.testing as npt

import riip


def test_create_catalog():
    rid = riip.RiiDataFrame()
    # dupl = ri.catalog.duplicated(subset=["book", "page"], keep=False)
    # if dupl.sum():
    #     print("Found duplication:")
    #     print(ri.catalog.loc[dupl, ["book", "page"]])
    #     raise Exception

    for _id in rid.catalog.query("'n' in tabulated").index:
        try:
            num_n = rid.catalog.loc[_id, "num_n"]
            wl_n = rid.raw_data.loc[_id, ["wl_n"]].to_numpy()[:num_n]
            n1 = rid.raw_data.loc[_id, ["n"]].to_numpy()[:num_n]
            n2 = riip.material.RiiMaterial(
                _id, rid.catalog, rid.raw_data, bound_check=False
            ).n(wl_n)
            eps = rid.material({"id": _id, "bound_check": False}).eps(wl_n)
            n3 = np.sqrt(eps).real
            npt.assert_array_almost_equal(n1, n2)
            npt.assert_array_almost_equal(
                np.abs(n1), n3
            )  # abs is necessary only for metamaterial
        except Exception as e:
            print(_id)
            raise e

    for _id in rid.catalog.query("'k' in tabulated").index:
        try:
            num_k = rid.catalog.loc[_id, "num_k"]
            wl_k = rid.raw_data.loc[_id, ["wl_k"]].to_numpy()[:num_k]
            k1 = rid.raw_data.loc[_id, ["k"]].to_numpy()[:num_k]
            k2 = riip.material.RiiMaterial(
                _id, rid.catalog, rid.raw_data, bound_check=False
            ).k(wl_k)
            eps = rid.material({"id": _id, "bound_check": False}).eps(wl_k)
            k3 = np.sqrt(eps).imag
            npt.assert_array_almost_equal(k1, k2)
            npt.assert_array_almost_equal(k1, k3)
        except Exception as e:
            print(_id)
            raise e
