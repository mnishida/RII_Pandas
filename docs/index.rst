RII_Pandas User's Guide
=======================
RII_Pandas
----------

RII_Pandas is Python 3 + `Pandas <https://pandas.pydata.org/>`_  wrapper for the `refractiveindex.info database <http://refractiveindex.info/>`_ developed by `Mikhail Polyanskiy <https://github.com/polyanskiy>`_.
It provides Pandas DataFrames for the catalog and experimental data stored in refractiveindex.info database based on the method of 'dboperations.py' from
`refractiveindex.info-sqlite package <https://github.com/HugoGuillen/refractiveindex.info-sqlite>`_ developed by `Hugo Guillén <https://github.com/HugoGuillen>`_.
This package will make it easy to access dielectric properties of various materials in a Python framework.

Features
^^^^^^^^
Create Pandas DataFrame by parsing database files cloned from Polyanskiy's `GitHub repository <https://github.com/polyanskiy/refractiveindex.info-database>`_.
Drude-Lorentz model (formula 21) and Brendel-Bormann model (formula 22) are availableto describe metallic dielectric function.

.. toctree::
    :glob:

    notebooks/*
