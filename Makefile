PYTHON_VERSION ?= 3.14
NUMPY_VERSION ?= 2.4.6
CONDA_BUILD_CONFIG := conda_pkg/conda_build_config.yaml
CONDA_DEV_PACKAGES := python=$(PYTHON_VERSION) numpy=$(NUMPY_VERSION) scipy cython setuptools pip conda-build
RUNTIME_PIP_PACKAGES := tables

dev-setup:
	conda install -y -c defaults $(CONDA_DEV_PACKAGES)
	pip install -e '.[dev,test,typecheck]'
	pre-commit install

CONDA_BLD_DIR := $(shell conda info --base)/conda-bld

$(CONDA_BUILD_CONFIG): Makefile
	@mkdir -p conda_pkg
	@printf "python:\n  - %s\nnumpy:\n  - %s\n" "$(PYTHON_VERSION)" "$(NUMPY_VERSION)" > $@

conda-build: $(CONDA_BUILD_CONFIG)
	conda build --no-test -c local -c mnishida -c defaults conda_pkg

conda-install: conda-build
	@PKG_PATH=$$(ls $(CONDA_BLD_DIR)/linux-64/riip-*.conda $(CONDA_BLD_DIR)/linux-64/riip-*.tar.bz2 $(CONDA_BLD_DIR)/noarch/riip-*.conda $(CONDA_BLD_DIR)/noarch/riip-*.tar.bz2 2>/dev/null | head -n 1); \
	if [ -n "$$PKG_PATH" ]; then \
		echo "Installing local package: $$PKG_PATH"; \
		conda install -y "$$PKG_PATH" --force-reinstall; \
		pip install $(RUNTIME_PIP_PACKAGES); \
	else \
		echo "Local package not found in $(CONDA_BLD_DIR)"; \
		exit 1; \
	fi

conda: conda-install

test:
	pytest

cov:
	pytest --cov riip

typecheck:
	pyrefly check .

lint:
	ruff check .

format:
	ruff format .

docs:
	pip install -e '.[docs]'
