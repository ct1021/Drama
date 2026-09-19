#!/usr/bin/env bash
# Explicit invocation installs a separate environment. Never starts training.
set -Eeuo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DRAMA_DATA_DIR:-/root/autodl-tmp/drama-research-20260919}"
ENV_DIR="${DRAMA_ENV_DIR:-$DATA_DIR/envs/drama-cu128}"
BASE_PYTHON="${DRAMA_BASE_PYTHON:-python}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
command -v nvcc >/dev/null || { echo 'CUDA development toolkit is required'; exit 1; }
command -v g++ >/dev/null || { echo 'C++ compiler is required'; exit 1; }
mkdir -p "$DATA_DIR/reports" "$DATA_DIR/cache/pip" "$DATA_DIR/cache/torch" "$DATA_DIR/cache/triton"
export PIP_CACHE_DIR="$DATA_DIR/cache/pip"
export TORCH_EXTENSIONS_DIR="$DATA_DIR/cache/torch"
export TRITON_CACHE_DIR="$DATA_DIR/cache/triton"
export MAX_JOBS="${MAX_JOBS:-8}"
export WANDB_MODE=disabled
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee "$DATA_DIR/reports/setup-$STAMP.log") 2>&1
if [[ -e "$ENV_DIR" && ! -f "$ENV_DIR/.drama-portable-env" ]]; then
    echo "Refusing to overwrite an unmarked environment: $ENV_DIR"
    exit 1
fi
if [[ ! -e "$ENV_DIR" ]]; then
    "$BASE_PYTHON" -m venv "$ENV_DIR"
    touch "$ENV_DIR/.drama-portable-env"
fi
PYTHON="$ENV_DIR/bin/python"
CONSTRAINTS="$REPO_DIR/ops/constraints-cu128.txt"
"$PYTHON" -m pip install --index-url https://pypi.org/simple 'pip==25.1.1' 'setuptools==75.8.0' wheel ninja packaging
"$PYTHON" -m pip install -c "$CONSTRAINTS" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install --index-url https://pypi.org/simple -c "$CONSTRAINTS" einops transformers
"$PYTHON" -m pip install --index-url https://pypi.org/simple -c "$CONSTRAINTS" causal-conv1d --no-build-isolation
"$PYTHON" -m pip install --index-url https://pypi.org/simple -c "$CONSTRAINTS" mamba-ssm --no-build-isolation
"$PYTHON" -m pip install --index-url https://pypi.org/simple -c "$CONSTRAINTS" -r "$REPO_DIR/requirements.txt"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$DATA_DIR/reports/freeze-$STAMP.txt"
cd "$REPO_DIR"
git rev-parse HEAD > "$DATA_DIR/reports/code-$STAMP.txt"
"$PYTHON" ops/probe.py > "$DATA_DIR/reports/probe-$STAMP.json"
"$PYTHON" ops/verify_runtime.py > "$DATA_DIR/reports/verify-$STAMP.json"
echo "Environment checks passed. Evidence: $DATA_DIR/reports. No training started."
