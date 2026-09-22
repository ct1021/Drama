"""Bounded engineering gate for one budget-matched candidate; no formal training."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'lightweight-routed-budgetmatched'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = dict(phase='running', pid=os.getpid(), code_commit=commit,
                  profile=PROFILE, started_unix=time.time(), formal_training_started=False)

    def save():
        temp = output / 'gate.tmp'
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(output / 'gate.json')

    def execute(command, name, limit):
        with (output / name).open('xb') as log:
            child = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s',
                                    str(limit)+'s', *command], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if child.returncode:
            raise RuntimeError(f'{name} failed: {child.returncode}')

    save()
    try:
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        assert shutil.disk_usage(output).free > 8*2**30
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
        execute([sys.executable, 'ops/test_experiment_profiles.py'], 'profiles.log', 60)
        execute([sys.executable, 'ops/test_task_routing.py'], 'routing.log', 60)
        execute([sys.executable, 'ops/verify_task_routing.py', '--profiles', PROFILE],
                'gpu-structure.log', 600)
        run = output / 'short-1800'
        run.mkdir()
        execute([sys.executable, '-u', 'ops/run_atari100k.py', '--run-dir', str(run),
                 '--profile', PROFILE, '--game', 'Boxing', '--seed', '3710', '--steps', '1800',
                 '--eval-episodes', '10', '--max-hours', '0.24'], 'short-training.log', 900)
        status = json.loads((run / 'status.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        assert status['phase'] == 'completed' and status['step'] == 1800
        assert manifest['code_commit'] == commit and manifest['profile'] == PROFILE
        assert all((run / 'ckpt' / name).stat().st_size > 0
                   for name in ('world_model-final.pth', 'agent-final.pth'))
        count = 0
        for line in (run / 'metrics.jsonl').open():
            assert math.isfinite(json.loads(line)['value'])
            count += 1
        assert count > 0
        result.update(phase='passed', finished_unix=time.time(), short_status=status,
                      finite_metrics=count, short_run=str(run),
                      claim='Engineering gate only; no RL performance claim')
        save()
    except BaseException as exc:
        result.update(phase='failed', error=repr(exc), finished_unix=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
