import numpy as np
import numpy.testing as npt

import riip
from riip.formulas import formulas_cython_dict, formulas_numpy_dict


def test_cython_formulas():
    rid = riip.RiiDataFrame()
    fbps = {
        1: ("MgAl2O4", "Tropf"),
        2: ("Ar", "Borzsonyi"),
        3: ("methanol", "Moutzouris"),
        4: ("BAB2O4", "Eimeri-o"),
        5: ("SIC", "Shaffer"),
        6: ("Ar", "Bideau-Mehu"),
        7: ("Si", "Edwards"),
        8: ("AgBr", "Schröter"),
        9: ("urea", "Rosker-e"),
        21: ("Ag", "Rakic-DLF"),
        22: ("Cu", "Rakic-BBF"),
    }

    for f, (b, p) in fbps.items():
        m = rid.material({"book": b, "page": p})
        wls = np.linspace(m.wl_max, m.wl_min)
        ws = 2 * np.pi / wls
        f_c = [formulas_cython_dict[f](w, m.cs) for w in ws]
        f_n = [formulas_numpy_dict[f](wl, m.cs) for wl in wls]
        npt.assert_allclose(f_c, f_n)
