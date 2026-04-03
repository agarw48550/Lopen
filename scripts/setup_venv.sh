#!/usr/bin/env bash
# setup_venv.sh — Create Python virtual environment and install dependencies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"

echo "==> Setting up Python virtual environment at $VENV_DIR"

# Prefer python3.11, fallback to python3
PYTHON_BIN=$(command -v python3.11 2>/dev/null || command -v python3 || echo "")
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3 not found. Run scripts/bootstrap.sh first."
    exit 1
fi

echo "--> Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "--> Virtual environment created at $VENV_DIR"
else
    echo "--> Virtual environment already exists, updating..."
fi

source "$VENV_DIR/bin/activate"

echo "--> Upgrading pip..."
pip install --upgrade pip wheel setuptools --quiet

echo "--> Installing requirements..."
pip install -r "$REPO_ROOT/requirements.txt"

echo "--> Installing Playwright browsers..."
playwright install chromium --with-deps 2>/dev/null || echo "    Playwright chromium install skipped (non-fatal)"

echo ""
echo "==> Setup complete!"
echo "    Activate with: source $VENV_DIR/bin/activate"
echo "    Then run:      bash scripts/start.sh"
