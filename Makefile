# Common tasks. Run `make help` for the list. (On Windows, run the commands shown here directly.)
PY ?= python

.PHONY: help install lint typecheck test pipeline check docker

help:
	@echo "install    install runtime + dev dependencies and the package (editable)"
	@echo "lint       ruff"
	@echo "typecheck  mypy on src/"
	@echo "test       pytest with coverage"
	@echo "pipeline   process + train (falls back to the bundled sample without collected data)"
	@echo "check      everything CI runs except the Docker build"
	@echo "docker     build the image"

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

lint:
	$(PY) -m ruff check src tests scripts

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest --cov

pipeline:
	$(PY) -m tram_mlops.process
	$(PY) -m tram_mlops.train

check: lint typecheck test pipeline

docker:
	docker build -t tram-mlops .
