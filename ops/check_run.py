"""Read-only snapshot of one run. Never launches, resumes, or stops training."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
run = args.run_dir.resolve()
status_path = run / 'status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {}
pid = status.get('pid')
cmdline_path = Path(f'/proc/{pid}/cmdline') if pid else None
command = cmdline_path.read_bytes().replace(b'\0', b' ').decode(errors='replace') if cmdline_path and cmdline_path.exists() else ''
# Require both the runner and this run directory, guarding against reused PIDs.
alive = 'ops/run_atari100k.py' in command and str(run) in command
evaluation_path = run / 'evaluations.jsonl'
evaluations = [json.loads(line) for line in evaluation_path.read_text().splitlines()] if evaluation_path.exists() else []
console = run / 'console.log'
tail = ''
if console.exists():
    with console.open('rb') as handle:
        handle.seek(max(0, console.stat().st_size - 8000))
        tail = handle.read().decode(errors='replace')
result = dict(checked_unix=time.time(), run_dir=str(run), process_alive=alive,
              status=status, status_age_seconds=time.time()-status.get('updated_unix', time.time()),
              evaluations=evaluations,
              checkpoints=[dict(name=p.name, bytes=p.stat().st_size, modified=p.stat().st_mtime)
                           for p in sorted((run/'ckpt').glob('*.pth'))],
              gpu=subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'],
                                 text=True, capture_output=True, timeout=15).stdout.strip(),
              console_tail=tail)
print(json.dumps(result, indent=2, ensure_ascii=False))
