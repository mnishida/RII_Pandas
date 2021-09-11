import os

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

ext_modules = []
e = Extension(
    "riip.formulas_cython",
    sources=[os.path.join("src", "utils", "formulas_cython.pyx")],
    depends=[],
    include_dirs=[np.get_include(), "."],
    language="c++",
)
e.cython_directives = {"language_level": "3"}
ext_modules.append(e)

setup(
    name="riip",
    version="0.5.0",
    url="https://github.com/mnishida/RII_Pandas",
    license="MIT",
    author="Munehiro Nishida",
    author_email="mnishida@hiroshima-u.ac.jp",
    description="Python 3 + Pandas wrapper for the refractiveindex.info database",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    packages=find_packages("src"),
    package_dir={"": "src"},
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
