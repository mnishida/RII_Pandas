install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt --upgrade
	pip install -r requirements_dev.txt --upgrade
	pip install -e . --upgrade
	pre-commit install

conda:
	conda install -c mnishida -c defaults -c conda-forge --file conda_pkg/conda_requirements_dev.txt
	conda install -c mnishida --file conda_pkg/conda_requirements.txt
	conda install -c mnishida pytables
	conda build --numpy 1.20 conda_pkg
	conda install --use-local --force-reinstall riip
	pre-commit install

test:
	pytest

cov:
	pytest --cov riip

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8

lintd2:
	flake8 --select RST

lintd:
	pydocstyle --convention google riip
