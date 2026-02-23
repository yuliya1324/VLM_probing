#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1) Create extract venv with Python 3.11
uv venv -p 3.11 "$REPO_ROOT/.venv-extract"

# 2) Activate venv
source "$REPO_ROOT/.venv-extract/bin/activate"

# 3) Install dependencies
pip install -U pip
pip install -r "$REPO_ROOT/requirements-extract.txt"

echo "[OK] Extract environment is ready: $REPO_ROOT/.venv-extract"