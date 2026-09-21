#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source ops/activate.sh
export CUDA_VISIBLE_DEVICES=0
RUN_DIR="$DRAMA_DATA_DIR/runs/boxing-lightweight-baseline-s3710-100k-20260921"
mkdir -p "$DRAMA_DATA_DIR/runs"
# An atomic directory creation prevents concurrent duplicate launches.
if ! mkdir "$RUN_DIR"; then
    echo 'Run directory exists; refusing duplicate launch.' >&2
    exit 1
fi
nohup timeout --signal=TERM --kill-after=60s 8h \
    python -u ops/run_atari100k.py --run-dir "$RUN_DIR" \
    --profile lightweight --game Boxing --seed 3710 --steps 100000 \
    --eval-episodes 10 --max-hours 7.9 \
    > "$RUN_DIR/console.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/launch.pid"
echo "Launched one lightweight baseline: $RUN_DIR (supervisor PID $(cat "$RUN_DIR/launch.pid"))"
