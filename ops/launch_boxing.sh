#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source ops/activate.sh
export CUDA_VISIBLE_DEVICES=0
RUN_DIR="$DRAMA_DATA_DIR/runs/boxing-baseline-s3710-100k-20260919"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/launch.pid" || -e "$RUN_DIR/status.json" ]]; then
    echo 'Run exists; refusing duplicate launch.' >&2
    exit 1
fi
# Timeout bounds this one run, including its evaluations. No automatic next job.
nohup timeout --signal=TERM --kill-after=60s 12h \
    python -u ops/run_atari100k.py --run-dir "$RUN_DIR" \
    --game Boxing --seed 3710 --steps 100000 --eval-episodes 10 --max-hours 11.9 \
    > "$RUN_DIR/console.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/launch.pid"
echo "Launched one run: $RUN_DIR (supervisor PID $(cat "$RUN_DIR/launch.pid"))"
