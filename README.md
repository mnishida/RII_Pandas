# RII_Pandas (refractiveindex.info-pandas)

[![PyPI version][pypi-image]][pypi-link]
[![Anaconda Version][anaconda-v-image]][anaconda-v-link]
[![Lint and Test][github-workflow-image]][github-workflow-link]

[pypi-image]: https://badge.fury.io/py/riip.svg
[pypi-link]: https://pypi.org/project/riip
[anaconda-v-image]: https://anaconda.org/mnishida/riip/badges/version.svg
[anaconda-v-link]: https://anaconda.org/mnishida/riip
[github-workflow-image]: https://github.com/mnishida/RII_Pandas/actions/workflows/pythonapp.yml/badge.svg
[github-workflow-link]: https://github.com/mnishida/RII_Pandas/actions/workflows/pythonapp.yml

Python 3 + [Pandas](https://pandas.pydata.org/) wrapper for the [refractiveindex.info database](http://refractiveindex.info/) developed by [Mikhail Polyanskiy](https://github.com/polyanskiy).

Pandas DataFrame creation was made with modified versions of `dboperations.py` from [refractiveindex.info-sqlite package](https://github.com/HugoGuillen/refractiveindex.info-sqlite) developed by [Hugo Guillén](https://github.com/HugoGuillen).

## Features
- Create Pandas DataFrame by parsing database files cloned from Polyanskiy's  [GitHub repository](https://github.com/polyanskiy/refractiveindex.info-database).
- Drude-Lorentz model (formula 21) and Brendel-Bormann model (formula 22) are available to describe metallic dielectric function.

## Install
#### Install and update using pip
```
$ pip install -U riip
```
#### Install using conda
```
$ conda install -c mnishida riip
```

## Usage
```
>>> import riip
>>> ri = riip.RiiDataFrame()
```
Polyanskiy's 'refractiveindex.info database' is cloned from [GitHub repository](https://github.com/polyanskiy/refractiveindex.info-database),
and three csv files, 'catalog.csv', 'raw_data.csv' and 'grid_data.csv' are created.
They are located in 'data' folder under the installation directory.
This process may take a few minutes, but it will happen only the first time you start it after installation.
```
>>> ri.search("Al")                      # search material contains the string
>>> ri.select(
      "2.5 < n < 3 and 0.4 < wl < 0.8"
    )                                    # select materials that fullfill the condition
>>> print(ri.show([23, 118]))            # show catalog
>>> print(ri.read(23))                   # read the data book
>>> ri.references(23)                    # see references
>>> ri.plot(23, "n")                     # plot wavelength dependence of refractive index

>>> Al = ri.material(
  {"book": "Al", "page": "Mathewson"})   # create material with book and page
>>> Al = ri.material({"id": 23})         # create material with id number
```

It may not be safe to use "id" in your application importing this package.
The id number may be changed when an update is done on your local database.

```
>>> import numpy as np
>>> wls = np.linspace(0.5, 1.6)          # wavelength from 0.5 μm to 1.6 μm
>>> Al.n(wls)                            # refractive index
>>> Al.k(wls)                            # extinction coefficient
>>> Al.eps(wls)                          # complex permittivity
```
For more information, see [RII_Pandas User's Guide](https://rii-pandas.readthedocs.io/en/latest/).

## Update database
If [refractiveindex.info database](http://refractiveindex.info/) is updated, you can pull it to the local database by

```
>>> import riip
>>> ri = riip.RiiDataFrame()
>>> ri.update_db()
```

## Uninstall
```
$ pip uninstall riip
```
or
```
$ conda uninstall riip
```

## Dependencies
- python 3
- numpy
- scipy
- pandas
- pyyaml
- gitpython

## Version
0.6.13
