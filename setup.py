import io
import os
import re

from setuptools import find_packages, setup
import riip


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name="riip",
    version=riip.__version__,
    url="https://github.com/mnishida/RII_Pandas",
    license=riip.__license__,
    author=riip.__author__,
    author_email="mnishida@hiroshima-u.ac.jp",
    description="Python 3 + Pandas wrapper for the refractiveindex.info database",
    long_description=read("README.md"),
    packages=find_packages(),
    install_requires=get_install_requires(),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
