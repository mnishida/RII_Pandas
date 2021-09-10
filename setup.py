import os
import sys

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

lib = os.path.join(sys.prefix, "lib")
include = os.path.join(sys.prefix, "include")


ext_modules = []
e = Extension(
    "riip.utils.formulas",
    sources=[os.path.join("riip", "utils", "formulas.pyx")],
    depends=[],
    include_dirs=[np.get_include(), include, "."],
    library_dirs=[lib],
    language="c++",
)
e.cython_directives = {"language_level": "3"}
ext_modules.append(e)

setup(
    name="riip",
    version="0.4.0",
    url="https://github.com/mnishida/RII_Pandas",
    license="MIT",
    author="Munehiro Nishida",
    author_email="mnishida@hiroshima-u.ac.jp",
    description="Python 3 + Pandas wrapper for the refractiveindex.info database",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=["Cython", "numpy", "scipy"],
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
