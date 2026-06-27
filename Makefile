PYTHON_VERSION ?= 3.14
NUMPY_VERSION ?= 2.4.6
PYREFLY_ENV ?= kkr-typecheck
CONDA_TARGET_ENV ?= $(CONDA_DEFAULT_ENV)
CONDA_ENV_ARGS := $(if $(CONDA_TARGET_ENV),-n $(CONDA_TARGET_ENV),)
PIP_INSTALL_CMD := $(if $(CONDA_TARGET_ENV),conda run -n $(CONDA_TARGET_ENV) python -m pip install,python -m pip install)
CONDA_BUILD_CONFIG := conda_pkg/conda_build_config.yaml
CONDA_DEV_PACKAGES := python=$(PYTHON_VERSION) numpy=$(NUMPY_VERSION) scipy cython setuptools pip conda-build
RUNTIME_PIP_PACKAGES := tables

dev-setup:
	conda install -y -c defaults $(CONDA_DEV_PACKAGES)
	pip install -e '.[dev,test,typecheck]'
	$(MAKE) typecheck-setup
	pre-commit install

CONDA_BLD_DIR := $(shell conda info --base)/conda-bld

$(CONDA_BUILD_CONFIG): Makefile
	@mkdir -p conda_pkg
	@printf "python:\n  - %s\nnumpy:\n  - %s\n" "$(PYTHON_VERSION)" "$(NUMPY_VERSION)" > $@

conda-build: $(CONDA_BUILD_CONFIG)
	conda build --no-test -c local -c mnishida -c defaults conda_pkg

conda-install: conda-build
	@if find "$(CONDA_BLD_DIR)" -maxdepth 2 -type f \( -name 'riip-*.conda' -o -name 'riip-*.tar.bz2' \) | grep -q .; then \
		echo "Installing local package: riip (channel: local)"; \
		conda remove $(CONDA_ENV_ARGS) -y riip >/dev/null 2>&1 || true; \
		conda install $(CONDA_ENV_ARGS) -y -c local -c mnishida -c defaults riip --force-reinstall; \
		$(PIP_INSTALL_CMD) $(RUNTIME_PIP_PACKAGES); \
	else \
		echo "Local package not found in $(CONDA_BLD_DIR)"; \
		exit 1; \
	fi

conda: conda-install

test:
	pytest

cov:
	pytest --cov riip

typecheck-setup:
	@conda run -n $(PYREFLY_ENV) python -c 'import sys; print(f"python-interpreter-path = \"{sys.executable}\"")' > pyrefly.toml

typecheck: typecheck-setup
	conda run -n $(PYREFLY_ENV) pyrefly check .

lint:
	ruff check .

format:
	ruff format .

docs:
	pip install -e '.[docs]'
