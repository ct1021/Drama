#!/usr/bin/env bash
# Explicit environment setup only. Does not launch training or overwrite an env.
set -Eeuo pipefail
if [[ $# -ne 2 ]]; then
    echo 'Usage: bash ops/install_restored_env.sh RESTORED_ARCHIVE_DIR NEW_DATA_DIR' >&2
    exit 2
fi
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESTORED_DIR="$(cd -- "$1" && pwd)"
mkdir -p "$2"
export DRAMA_DATA_DIR="$(cd -- "$2" && pwd)"
export DRAMA_ENV_DIR="$DRAMA_DATA_DIR/envs/drama-cu128"
export DRAMA_BASE_PYTHON="${DRAMA_BASE_PYTHON:-python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -e "$DRAMA_ENV_DIR" ]]; then
    echo 'Target environment already exists; choose a new data directory.' >&2
    exit 1
fi
"$DRAMA_BASE_PYTHON" -c 'import sys,platform; assert sys.version_info[:2] == (3,12); assert platform.system()=="Linux" and platform.machine()=="x86_64"'
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 5090' || {
    echo 'This recorded profile targets RTX 5090; other GPUs need separate validation.' >&2
    exit 1
}
"$DRAMA_BASE_PYTHON" - "$RESTORED_DIR" "$DRAMA_DATA_DIR" <<'PY'
import hashlib,json,pathlib,shutil,sys
src,dest=map(pathlib.Path,sys.argv[1:])
assert (src/'RESTORE-VERIFIED.json').is_file(), 'Run restore_experiment.py first'
manifest=json.loads((src/'MANIFEST.json').read_text())
records={x['path']:x for x in manifest['files']}
names=['environment/versions.txt','environment/wheels/causal_conv1d-1.5.2+sm120-cp312-cp312-linux_x86_64.whl']
for name in names:
    with (src/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==records[name]['sha256']
(dest/'built-wheels').mkdir(parents=True,exist_ok=True)
shutil.copy2(src/names[1],dest/'built-wheels'/pathlib.Path(names[1]).name)
PY
export PIP_CONSTRAINT="$RESTORED_DIR/environment/versions.txt"
export WANDB_MODE=disabled
"$DRAMA_BASE_PYTHON" "$REPO_DIR/ops/fetch_pinned_wheels.py" "$DRAMA_DATA_DIR/wheels"
bash "$REPO_DIR/ops/bootstrap.sh"
echo 'Environment restored and checked. No training has been started.'
