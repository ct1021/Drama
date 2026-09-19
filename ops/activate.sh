# Source this file: source /path/to/Drama/ops/activate.sh
# Does not install packages or start any process.
DRAMA_DATA_DIR="${DRAMA_DATA_DIR:-/root/autodl-tmp/drama-research-20260919}"
DRAMA_ENV_DIR="${DRAMA_ENV_DIR:-$DRAMA_DATA_DIR/envs/drama-cu128}"
if [[ ! -f "$DRAMA_ENV_DIR/bin/activate" ]]; then
    echo "Environment missing: $DRAMA_ENV_DIR" >&2
    return 1
fi
source "$DRAMA_ENV_DIR/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export PIP_CACHE_DIR="$DRAMA_DATA_DIR/cache/pip"
export TORCH_EXTENSIONS_DIR="$DRAMA_DATA_DIR/cache/torch"
export TRITON_CACHE_DIR="$DRAMA_DATA_DIR/cache/triton"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
