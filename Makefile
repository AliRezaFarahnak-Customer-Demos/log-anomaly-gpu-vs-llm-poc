PY ?= .venv/bin/python
export ML_ROOT ?= ./.ml
DATA ?= data/synthetic/v1
CONFIG ?= configs/tiny.yaml
SPLIT ?= test
CONTEXT ?= expected-flow
RUN_ID ?=
IMAGE ?= logpoc:local

.PHONY: help install data data-prod-like test lint build train resume evaluate poc2 baseline compare

help:
	@echo "targets: install data data-prod-like test lint build train resume evaluate poc2 baseline compare"

install:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e ".[dev,train,llm,mlflow]"

data:
	$(PY) -m logpoc generate --config configs/data_v1.yaml --out $(DATA)
	$(PY) -m logpoc prepare --data $(DATA)

test:
	$(PY) -m ruff check .
	$(PY) -m pytest

data-prod-like:
	$(PY) -m logpoc generate-prod-like --config configs/data_prod_like.yaml --out data/synthetic/prod_like

lint:
	$(PY) -m ruff check .

build:
	docker build -t $(IMAGE) --build-arg GIT_SHA=$$(git rev-parse --short HEAD) \
		--build-arg GIT_DIRTY=$$([ -n "$$(git status --porcelain)" ] && echo true || echo false) .

# Local training. Needs the base model under $(ML_ROOT)/models (see README quick start).
train:
	RUN_ID=$(RUN_ID) CONFIG=$(CONFIG) $(PY) -m logpoc train --data $(DATA)

# Same command as train: with the same RUN_ID it resumes from the newest complete checkpoint.
resume: train

evaluate:
	$(PY) -m logpoc evaluate --run-id $(RUN_ID) --data $(DATA) --split $(SPLIT)

baseline:
	$(PY) -m logpoc baseline-grep --data $(DATA) --split $(SPLIT)

poc2:
	$(PY) -m logpoc poc2-llm --data $(DATA) --split $(SPLIT) --context $(CONTEXT) --dry-run

compare:
	$(PY) -m logpoc compare
