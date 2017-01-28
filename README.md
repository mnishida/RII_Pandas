# RII_Pandas (refractiveindex.info-pandas)
Python 3 + Pandas wrapper for the [refractiveindex.info database](http://refractiveindex.info/) by [Mikhail Polyanskiy](https://github.com/polyanskiy).

Pandas DataFrame creation was made with modified versions of `dboperations.py` from [refractiveindex.info-sqlite package](https://github.com/HugoGuillen/refractiveindex.info-sqlite) by [Hugo Guillén](https://github.com/HugoGuillen).

##Features
- Create Pandas DataFrame by parsing yml files cloned from [GitHub repository](https://github.com/polyanskiy/refractiveindex.info-database).

## Usage

` $ git clone https://github.com/mnishida/RII_Pandas.git`  
` $ cd RII_Pandas`  
` $ python setup.py develop`  

## Uninstall

` $ pip uninstall riip`  

## Dependencies
- python 3
- numpy
- scipy
- pandas
- pyyaml
- gitpython

## Disclaimer
Same as the refractiveindex.info webpage: *NO GUARANTEE OF ACCURACY - Use on your own risk*.

## Version
2017-01-25
