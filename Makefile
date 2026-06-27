PYTHON_VERSION ?= 3.14
NUMPY_VERSION ?= 2.4.6
PYREFLY_ENV ?= kkr-typecheck
# Keep conda behavior stable even when shell startup leaks CONDA_ENVS_PATH.
CONDA_EXEC := env -u CONDA_ENVS_PATH conda
CONDA_ROOT_PREFIX := $(shell if [ -n "$$CONDA_EXE" ]; then dirname "$$(dirname "$$CONDA_EXE")"; else env -u CONDA_ENVS_PATH conda info --base; fi)
CONDA_ACTIVE_ENV_FROM_INFO := $(shell $(CONDA_EXEC) info --json 2>/dev/null | python -c 'import json,sys; print((json.load(sys.stdin).get("active_prefix_name") or ""))' 2>/dev/null)
CONDA_ACTIVE_ENV_FROM_PREFIX := $(notdir $(CONDA_PREFIX))
CONDA_TARGET_ENV ?= $(if $(CONDA_DEFAULT_ENV),$(CONDA_DEFAULT_ENV),$(if $(CONDA_ACTIVE_ENV_FROM_PREFIX),$(CONDA_ACTIVE_ENV_FROM_PREFIX),$(CONDA_ACTIVE_ENV_FROM_INFO)))
# Support both env names and absolute prefixes.
CONDA_TARGET_ENV_ARGS := $(if $(CONDA_TARGET_ENV),$(if $(filter /%,$(CONDA_TARGET_ENV)),-p $(CONDA_TARGET_ENV),-n $(CONDA_TARGET_ENV)),)
CONDA_RUN_TARGET := $(CONDA_EXEC) run $(CONDA_TARGET_ENV_ARGS)
PYREFLY_ENV_PREFIX := $(CONDA_ROOT_PREFIX)/envs/$(PYREFLY_ENV)
PYREFLY_ENV_ARGS := -p $(PYREFLY_ENV_PREFIX)
CONDA_RUN_PYREFLY := $(CONDA_EXEC) run $(PYREFLY_ENV_ARGS)
# Pre-remove mode before conda install:
# - none  : keep existing environment graph (default)
# - force : remove only the target package with dependency checks disabled
CONDA_PRE_REMOVE_MODE ?= none
PIP_INSTALL_CMD := $(if $(CONDA_TARGET_ENV),$(CONDA_RUN_TARGET) python -m pip install,python -m pip install)
PIP_INSTALL_NO_DEPS_CMD := $(if $(CONDA_TARGET_ENV),$(CONDA_RUN_TARGET) python -m pip install --no-deps,pip install --no-deps)
PYTEST_RUN_CMD := $(if $(CONDA_TARGET_ENV),$(CONDA_RUN_TARGET) pytest,pytest)
CONDA_BUILD_CONFIG := conda_pkg/conda_build_config.yaml
CONDA_DEV_PACKAGES := python=$(PYTHON_VERSION) numpy=$(NUMPY_VERSION) scipy cython setuptools pip conda-build
TYPECHECK_CONDA_PACKAGES := python=$(PYTHON_VERSION) pip
DEV_PIP_PACKAGES ?= $(shell python -c 'import tomllib,pathlib; d=tomllib.loads(pathlib.Path("pyproject.toml").read_text()); print(" ".join(d.get("project",{}).get("optional-dependencies",{}).get("dev",[])))')
TYPECHECK_PIP_PACKAGES ?= $(shell python -c 'import tomllib,pathlib; d=tomllib.loads(pathlib.Path("pyproject.toml").read_text()); print(" ".join(d.get("project",{}).get("optional-dependencies",{}).get("typecheck",[])))')
RUNTIME_PIP_PACKAGES := tables pytest-regressions

dev-setup:
	$(CONDA_EXEC) install -y -c defaults $(CONDA_DEV_PACKAGES)
	pip install -e '.[dev]'
	pre-commit install

dev-sync:
	$(MAKE) conda-install
	$(PIP_INSTALL_NO_DEPS_CMD) -e .

typecheck-env-setup:
	$(MAKE) conda-install CONDA_TARGET_ENV=$(PYREFLY_ENV_PREFIX)
	$(CONDA_EXEC) install $(PYREFLY_ENV_ARGS) -y -c defaults $(CONDA_DEV_PACKAGES)
	$(CONDA_EXEC) install $(PYREFLY_ENV_ARGS) -y -c defaults $(TYPECHECK_CONDA_PACKAGES)
	@if [ -n "$(strip $(DEV_PIP_PACKAGES))" ]; then \
		$(CONDA_RUN_PYREFLY) python -m pip install $(DEV_PIP_PACKAGES); \
	fi
	@if [ -n "$(strip $(TYPECHECK_PIP_PACKAGES))" ]; then \
		$(CONDA_RUN_PYREFLY) python -m pip install $(TYPECHECK_PIP_PACKAGES); \
	else \
		echo "No typecheck packages found in pyproject.toml [project.optional-dependencies].typecheck"; \
		exit 1; \
	fi
	$(MAKE) typecheck-setup

CONDA_BLD_DIR := $(shell $(CONDA_EXEC) info --base)/conda-bld

$(CONDA_BUILD_CONFIG): Makefile
	@mkdir -p conda_pkg
	@printf "python:\n  - %s\nnumpy:\n  - %s\n" "$(PYTHON_VERSION)" "$(NUMPY_VERSION)" > $@

conda-build: $(CONDA_BUILD_CONFIG)
	@tmp_src="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_src"' EXIT; \
	tar --exclude='./.git' -cf - . | tar -xf - -C "$$tmp_src"; \
	cd "$$tmp_src" && $(CONDA_EXEC) build --no-test -c local -c mnishida -c defaults conda_pkg

conda-install: conda-build
	@if find "$(CONDA_BLD_DIR)" -maxdepth 2 -type f \( -name 'riip-*.conda' -o -name 'riip-*.tar.bz2' \) | grep -q .; then \
		echo "Installing local package: riip (channel: local)"; \
		if [ "$(CONDA_PRE_REMOVE_MODE)" = "force" ]; then $(CONDA_EXEC) remove $(CONDA_TARGET_ENV_ARGS) -y --force riip >/dev/null 2>&1 || true; fi; \
		$(CONDA_EXEC) install $(CONDA_TARGET_ENV_ARGS) -y -c local -c mnishida -c defaults riip --force-reinstall --update-deps; \
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
	@$(CONDA_RUN_PYREFLY) python -c 'import sys; print(f"python-interpreter-path = \"{sys.executable}\"")' > pyrefly.toml

typecheck: typecheck-setup
	$(CONDA_RUN_PYREFLY) pyrefly check .

lint:
	ruff check .

format:
	ruff format .

docs:
	pip install -e '.[docs]'
