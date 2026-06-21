dev-setup:
	conda install -y -c defaults python=3.13 numpy=2.2.5 scipy cython setuptools pip conda-build anaconda-client
	pip install -e .[dev]
	pre-commit install

CONDA_BLD_DIR := $(shell conda info --base)/conda-bld

conda-build:
	conda build --no-test -c local -c mnishida -c defaults --numpy 2.2.5 conda_pkg

conda-install: conda-build
	@PKG_PATH=$$(ls $(CONDA_BLD_DIR)/linux-64/riip-*.conda $(CONDA_BLD_DIR)/linux-64/riip-*.tar.bz2 $(CONDA_BLD_DIR)/noarch/riip-*.conda $(CONDA_BLD_DIR)/noarch/riip-*.tar.bz2 2>/dev/null | head -n 1); \
	if [ -n "$$PKG_PATH" ]; then \
		echo "Installing local package: $$PKG_PATH"; \
		conda install -y "$$PKG_PATH" --force-reinstall; \
		pip install tables gitpython; \
	else \
		echo "Local package not found in $(CONDA_BLD_DIR)"; \
		exit 1; \
	fi

conda: conda-install

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
