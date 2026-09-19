"""Explicit migration step: fetch official runtime wheels and verify their hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('wheel_dir', type=Path)
args = parser.parse_args()
args.wheel_dir.mkdir(parents=True, exist_ok=True)
ops = Path(__file__).resolve().parent
manifest = json.loads((ops / 'wheels-cu128.json').read_text())
for item in manifest['assets']:
    # The portable archive already contains the tested local SM120 convolution.
    if item['filename'].startswith('causal_conv1d-'):
        continue
    target = args.wheel_dir / item['filename']
    if not target.exists():
        if item['filename'].startswith('torch-'):
            subprocess.run([sys.executable, str(ops/'download_wheel.py'),
                            item['url'], item['sha256'], str(args.wheel_dir)], check=True)
        else:
            temporary = target.with_suffix(target.suffix + '.download')
            subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                            '--connect-timeout', '15', '--max-time', '900',
                            '--output', str(temporary), item['url']], check=True)
            with temporary.open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError('Wheel hash mismatch: ' + item['filename'])
            temporary.replace(target)
    with target.open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != item['sha256']:
            raise ValueError('Cached wheel hash mismatch: ' + item['filename'])
    print('Verified:', target.name, flush=True)
