import io
import os
import re

from setuptools import find_packages, setup


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
    version="0.1.0",
    url="https://github.com/mnishida/RII_Pandas",
    license="MIT",
    author="Munehiro Nishida",
    author_email="mnishida@hiroshima-u.ac.jp",
    description="Python 3 + Pandas wrapper for the refractiveindex.info database",
    long_description=read("README.md"),
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_install_requires(),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
