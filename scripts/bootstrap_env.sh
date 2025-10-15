#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/bootstrap_env.sh              # CPU wheels
#   bash scripts/bootstrap_env.sh cu126        # CUDA 12.6 wheels
#
# Forces Python 3.11 venv (PyTorch 2.8 wheels support 3.11/3.12; avoid 3.13).

CUDA_TAG="${1:-cpu}"
PYBIN="${PYBIN:-python3.11}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "FATAL: $PYBIN not found. Install Python 3.11 or set PYBIN to a 3.11 interpreter." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[create] .venv with $PYBIN"
  "$PYBIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# Ensure a minimal package exists so hatchling can build editable wheels
if [ ! -d "mondeq" ]; then
  echo "[scaffold] mondeq/"
  mkdir -p mondeq
  printf "__all__ = []\n__version__ = '0.0.1'\n" > mondeq/__init__.py
fi

if [ "$CUDA_TAG" = "cpu" ]; then
  EXTRA_INDEX="https://download.pytorch.org/whl/cpu"
else
  EXTRA_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# Install our package (editable) + deps from pyproject.toml
pip install -e ".[dev,vision]" --extra-index-url "${EXTRA_INDEX}"

# Quick sanity check
python - <<'PY'
import sys, platform
import torch, numpy, scipy, matplotlib
msg = (
    f"[OK] python={platform.python_version()} "
    f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"numpy={__import__('numpy').__version__} scipy={scipy.__version__} mpl={matplotlib.__version__}"
)
print(msg, file=sys.stderr)
PY

echo "[done] environment ready."
