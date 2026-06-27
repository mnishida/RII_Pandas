PYTHON_VERSION ?= 3.14
NUMPY_VERSION ?= 2.4.6
PYREFLY_ENV ?= kkr-typecheck
CONDA_ACTIVE_ENV_FROM_INFO := $(shell conda info --json 2>/dev/null | python -c 'import json,sys; print((json.load(sys.stdin).get("active_prefix_name") or ""))' 2>/dev/null)
CONDA_ACTIVE_ENV_FROM_PREFIX := $(notdir $(CONDA_PREFIX))
CONDA_TARGET_ENV ?= $(if $(CONDA_DEFAULT_ENV),$(CONDA_DEFAULT_ENV),$(if $(CONDA_ACTIVE_ENV_FROM_PREFIX),$(CONDA_ACTIVE_ENV_FROM_PREFIX),$(CONDA_ACTIVE_ENV_FROM_INFO)))
CONDA_ENV_ARGS := $(if $(CONDA_TARGET_ENV),-n $(CONDA_TARGET_ENV),)
# Pre-remove mode before conda install:
# - none  : keep existing environment graph (default)
# - force : remove only the target package with dependency checks disabled
CONDA_PRE_REMOVE_MODE ?= none
PIP_INSTALL_CMD := $(if $(CONDA_TARGET_ENV),conda run -n $(CONDA_TARGET_ENV) python -m pip install,python -m pip install)
PIP_INSTALL_NO_DEPS_CMD := $(if $(CONDA_TARGET_ENV),conda run -n $(CONDA_TARGET_ENV) python -m pip install --no-deps,pip install --no-deps)
PYTEST_RUN_CMD := $(if $(CONDA_TARGET_ENV),conda run -n $(CONDA_TARGET_ENV) pytest,pytest)
CONDA_BUILD_CONFIG := conda_pkg/conda_build_config.yaml
CONDA_DEV_PACKAGES := python=$(PYTHON_VERSION) numpy=$(NUMPY_VERSION) scipy cython setuptools pip conda-build
TYPECHECK_CONDA_PACKAGES := python=$(PYTHON_VERSION) pip
DEV_PIP_PACKAGES ?= $(shell python -c 'import tomllib,pathlib; d=tomllib.loads(pathlib.Path("pyproject.toml").read_text()); print(" ".join(d.get("project",{}).get("optional-dependencies",{}).get("dev",[])))')
TYPECHECK_PIP_PACKAGES ?= $(shell python -c 'import tomllib,pathlib; d=tomllib.loads(pathlib.Path("pyproject.toml").read_text()); print(" ".join(d.get("project",{}).get("optional-dependencies",{}).get("typecheck",[])))')
RUNTIME_PIP_PACKAGES := tables pytest-regressions

dev-setup:
	conda install -y -c defaults $(CONDA_DEV_PACKAGES)
	pip install -e '.[dev]'
	pre-commit install

dev-sync:
	$(MAKE) conda-install
	$(PIP_INSTALL_NO_DEPS_CMD) -e .

typecheck-env-setup:
	$(MAKE) conda-install CONDA_TARGET_ENV=$(PYREFLY_ENV)
	conda install -n $(PYREFLY_ENV) -y -c defaults $(CONDA_DEV_PACKAGES)
	conda install -n $(PYREFLY_ENV) -y -c defaults $(TYPECHECK_CONDA_PACKAGES)
	@if [ -n "$(strip $(DEV_PIP_PACKAGES))" ]; then \
		conda run -n $(PYREFLY_ENV) python -m pip install $(DEV_PIP_PACKAGES); \
	fi
	@if [ -n "$(strip $(TYPECHECK_PIP_PACKAGES))" ]; then \
		conda run -n $(PYREFLY_ENV) python -m pip install $(TYPECHECK_PIP_PACKAGES); \
	else \
		echo "No typecheck packages found in pyproject.toml [project.optional-dependencies].typecheck"; \
		exit 1; \
	fi
	$(MAKE) typecheck-setup

CONDA_BLD_DIR := $(shell conda info --base)/conda-bld

$(CONDA_BUILD_CONFIG): Makefile
	@mkdir -p conda_pkg
	@printf "python:\n  - %s\nnumpy:\n  - %s\n" "$(PYTHON_VERSION)" "$(NUMPY_VERSION)" > $@

conda-build: $(CONDA_BUILD_CONFIG)
	@tmp_src="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_src"' EXIT; \
	tar --exclude='./.git' -cf - . | tar -xf - -C "$$tmp_src"; \
	cd "$$tmp_src" && conda build --no-test -c local -c mnishida -c defaults conda_pkg

conda-install: conda-build
	@if find "$(CONDA_BLD_DIR)" -maxdepth 2 -type f \( -name 'riip-*.conda' -o -name 'riip-*.tar.bz2' \) | grep -q .; then \
		echo "Installing local package: riip (channel: local)"; \
		if [ "$(CONDA_PRE_REMOVE_MODE)" = "force" ]; then conda remove $(CONDA_ENV_ARGS) -y --force riip >/dev/null 2>&1 || true; fi; \
		conda install $(CONDA_ENV_ARGS) -y -c local -c mnishida -c defaults riip --force-reinstall --update-deps; \
		$(PIP_INSTALL_CMD) $(RUNTIME_PIP_PACKAGES); \
	else \
		echo "Local package not found in $(CONDA_BLD_DIR)"; \
		exit 1; \
	fi

conda: conda-install

test:
	$(PYTEST_RUN_CMD)

cov:
	$(PYTEST_RUN_CMD) --cov riip

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
