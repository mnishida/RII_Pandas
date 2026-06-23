import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = []
e = Extension(
    "riip.formulas_cython",
    sources=[os.path.join("src", "utils", "formulas_cython.pyx")],
    depends=[],
    include_dirs=[np.get_include(), "."],
    language="c++",
)
ext_modules.append(e)

setup(ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}), )
