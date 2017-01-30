# RII_Pandas (refractiveindex.info-pandas)
Python 3 + Pandas wrapper for the [refractiveindex.info database](http://refractiveindex.info/) by [Mikhail Polyanskiy](https://github.com/polyanskiy).

Pandas DataFrame creation was made with modified versions of `dboperations.py` from [refractiveindex.info-sqlite package](https://github.com/HugoGuillen/refractiveindex.info-sqlite) by [Hugo Guillén](https://github.com/HugoGuillen).

## Features
- Create Pandas DataFrame by parsing yml files cloned from [GitHub repository](https://github.com/polyanskiy/refractiveindex.info-database).

## Install

```
git clone https://github.com/mnishida/RII_Pandas.git
cd RII_Pandas
python setup.py develop
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
pip uninstall riip
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
