#!/usr/bin/env bash
# One-time setup: creates a Python 3.12 virtualenv and installs everything.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python toolchain manager)…"
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi

echo "Creating Python 3.12 virtualenv in .venv …"
uv venv --python 3.12 .venv

echo "Installing dependencies (this downloads PyTorch — a few hundred MB)…"
uv pip install --python .venv/bin/python -r requirements.txt

echo
echo "✅  Setup complete.  Start it with:   ./run.sh"
