#!/usr/bin/env bash
# Runs once after the dev container is created.
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y -qq shellcheck >/dev/null

pip install --quiet uv
uv venv --python 3.12 --clear .venv
uv pip install --python .venv/bin/python -e ".[dev,train,llm,mlflow]"

echo "dev container ready. Run: make test"
