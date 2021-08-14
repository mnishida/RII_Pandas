install:
	pip install -r requirements.txt --upgrade
	pip install -r requirements_dev.txt --upgrade
	pip install -e .
	pre-commit install

conda:
	conda config --append channels anaconda
	conda config --append channels conda-forge
	conda install --file conda_requirements.txt
	conda install --file conda_requirements_dev.txt
	pip install -e .
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

doc8:
	doc8 docs/
