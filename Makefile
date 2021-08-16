install:
	pip install -r requirements.txt --upgrade
	pip install -r requirements_dev.txt --upgrade
	pip install -e .
	pre-commit install

conda:
	conda install numpy scipy pandas pyyaml gitpython matplotlib
	conda install sphinx==3.1.2 sphinx_rtd_theme recommonmark pytest flake8 black pydocstyle
	conda install -c conda-forge pre-commit tox pytest-regressions doc8 sphinx-markdown-tables sphinx-autodoc-typehints
	python setup.py develop
	pre-commit install

test:
	pytest

cov:
	pytest --cov= riip

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8

pylint:
	pylint riip

lintd2:
	flake8 --select RST

lintd:
	pydocstyle riip
