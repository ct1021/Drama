#!/usr/bin/env bash
# Explicit invocation installs a separate environment. Never starts training.
set -Eeuo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DRAMA_DATA_DIR:-/root/autodl-tmp/drama-research-20260919}"
ENV_DIR="${DRAMA_ENV_DIR:-$DATA_DIR/envs/drama-cu128}"
BASE_PYTHON="${DRAMA_BASE_PYTHON:-python}"
PYPI_INDEX="${DRAMA_PYPI_INDEX:-https://pypi.org/simple}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
command -v nvcc >/dev/null || { echo 'CUDA development toolkit is required'; exit 1; }
command -v g++ >/dev/null || { echo 'C++ compiler is required'; exit 1; }
mkdir -p "$DATA_DIR/reports" "$DATA_DIR/cache/pip" "$DATA_DIR/cache/torch" "$DATA_DIR/cache/triton"
export PIP_CACHE_DIR="$DATA_DIR/cache/pip"
export TORCH_EXTENSIONS_DIR="$DATA_DIR/cache/torch"
export TRITON_CACHE_DIR="$DATA_DIR/cache/triton"
export MAX_JOBS="${MAX_JOBS:-8}"
export WANDB_MODE=disabled
if [[ -d "$DATA_DIR/wheels" ]]; then
    export PIP_FIND_LINKS="$DATA_DIR/wheels"
fi
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
"$PYTHON" "$REPO_DIR/ops/prepare_wheels.py" "$DATA_DIR/wheels"
"$PYTHON" -m pip install --index-url "$PYPI_INDEX" 'pip==25.1.1' 'setuptools==75.8.0' wheel ninja packaging
# Fetch ordinary Python dependencies through the selected registry; the CUDA
# index can redirect these files to a slow overseas CDN on some cloud hosts.
"$PYTHON" -m pip install --index-url "$PYPI_INDEX" -c "$CONSTRAINTS" \
    filelock typing-extensions sympy networkx jinja2 fsspec numpy pillow mpmath MarkupSafe
TORCH_PACKAGES=(torch torchvision torchaudio)
TORCH_WHEEL="$DATA_DIR/wheels/torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
TRITON_WHEEL="$DATA_DIR/wheels/triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
if [[ -f "$TORCH_WHEEL" ]]; then
    TORCH_PACKAGES[0]="$TORCH_WHEEL"
fi
if [[ -f "$TRITON_WHEEL" ]]; then
    TORCH_PACKAGES+=("$TRITON_WHEEL")
fi
"$PYTHON" -m pip install -c "$CONSTRAINTS" "${TORCH_PACKAGES[@]}" --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install --index-url "$PYPI_INDEX" -c "$CONSTRAINTS" einops transformers
CAUSAL_PACKAGE=causal-conv1d
MAMBA_PACKAGE=mamba-ssm
if [[ -f "$DATA_DIR/wheels/causal_conv1d-1.5.2-cp312-cp312-linux_x86_64.whl" ]]; then
    CAUSAL_PACKAGE="$DATA_DIR/wheels/causal_conv1d-1.5.2-cp312-cp312-linux_x86_64.whl"
fi
if [[ -f "$DATA_DIR/wheels/mamba_ssm-2.2.6.post3-cp312-cp312-linux_x86_64.whl" ]]; then
    MAMBA_PACKAGE="$DATA_DIR/wheels/mamba_ssm-2.2.6.post3-cp312-cp312-linux_x86_64.whl"
fi
GPU_CAPABILITY="$("$PYTHON" -c 'import torch; print("%d%d" % torch.cuda.get_device_capability(0))')"
if [[ "$GPU_CAPABILITY" == "120" ]]; then
    bash "$REPO_DIR/ops/build_causal_sm120.sh"
else
    "$PYTHON" -m pip install --index-url "$PYPI_INDEX" -c "$CONSTRAINTS" "$CAUSAL_PACKAGE" --no-build-isolation
fi
"$PYTHON" -m pip install --index-url "$PYPI_INDEX" -c "$CONSTRAINTS" "$MAMBA_PACKAGE" --no-build-isolation
"$PYTHON" -m pip install --index-url "$PYPI_INDEX" -c "$CONSTRAINTS" -r "$REPO_DIR/requirements.txt"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$DATA_DIR/reports/freeze-$STAMP.txt"
cd "$REPO_DIR"
git rev-parse HEAD > "$DATA_DIR/reports/code-$STAMP.txt"
"$PYTHON" ops/probe.py > "$DATA_DIR/reports/probe-$STAMP.json"
"$PYTHON" ops/verify_runtime.py > "$DATA_DIR/reports/verify-$STAMP.json"
echo "Environment checks passed. Evidence: $DATA_DIR/reports. No training started."
