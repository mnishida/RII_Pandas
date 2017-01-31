# RII_Pandas (refractiveindex.info-pandas)
Python 3 + Pandas wrapper for the [refractiveindex.info database](http://refractiveindex.info/) developed by [Mikhail Polyanskiy](https://github.com/polyanskiy).

Pandas DataFrame creation was made with modified versions of `dboperations.py` from [refractiveindex.info-sqlite package](https://github.com/HugoGuillen/refractiveindex.info-sqlite) developed by [Hugo Guillén](https://github.com/HugoGuillen).

## Features
- Create Pandas DataFrame by parsing yml files cloned from Polyanskiy's  [GitHub repository](https://github.com/polyanskiy/refractiveindex.info-database).
- Drude-Lorentz model (formula 21) and Brendel-Bormann model (formula 22) are available to describe metal dielectric function.


## Install
I highly recommend to install [Anaconda](https://www.continuum.io/downloads) that has all required python packages.

#### For linux
Provided that Anaconda Python 3 is installed and activated,
```
$ conda create -n py36 python=3.6 numpy scipy pandas pyyaml gitpython notebook
$ source activate py36
$ git clone https://github.com/mnishida/RII_Pandas.git
$ python RII_Pandas/setup.py install
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
 >>> ri.catalog.loc[:30, ['book', 'page']]
 >>> ri.raw_data.loc[ri.raw_data['id']==10, ['wl_n', 'n']]  
 >>> ri.raw_data.loc[ri.raw_data['id']==10, ['wl_k', 'k']]
 >>> ri.grid_data.loc[ri.grid_data['id']==10, ['wl', 'n', 'k']]
```

## Update database
```
>>> import riip
>>> ri = riip.RiiDataFrame()  
>>> ri.update_db()  
```

## Uninstall
```
$ pip uninstall riip
```

## Dependencies
- python 3
- numpy
- scipy
- pandas
- pyyaml
- gitpython

## Version
0.1.0
