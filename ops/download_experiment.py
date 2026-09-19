"""Download a versioned experiment archive in verified, reusable parts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def valid_filename(name):
    if not name or name in {'.', '..'} or any(c in name for c in '/\\:'):
        raise ValueError('Unsafe filename: ' + name)


def download(manifest_path, directory):
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    directory.mkdir(parents=True, exist_ok=True)
    valid_filename(data['archive']['filename'])
    target = directory / data['archive']['filename']
    if target.exists():
        if target.stat().st_size != data['archive']['bytes'] or digest(target) != data['archive']['sha256']:
            raise ValueError('Existing archive does not match manifest')
        return target
    parts_dir = directory / 'parts'
    parts_dir.mkdir(exist_ok=True)
    parts = sorted(data['parts'], key=lambda p: p['index'])
    if [p['index'] for p in parts] != list(range(len(parts))):
        raise ValueError('Invalid part ordering')
    if not parts or len({p['name'] for p in parts}) != len(parts):
        raise ValueError('Empty or duplicate parts')
    for part in parts:
        valid_filename(part['name'])

    def fetch(part):
        url = urlparse(part['url'])
        if url.scheme != 'https' or url.hostname != 'github.com':
            raise ValueError('Expected a GitHub HTTPS release asset')
        path = parts_dir / part['name']
        if path.exists() and path.stat().st_size == part['bytes'] and digest(path) == part['sha256']:
            return path
        temporary = path.with_suffix(path.suffix + '.download')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(part['url'], timeout=60) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
                break
            except (OSError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if temporary.stat().st_size != part['bytes'] or digest(temporary) != part['sha256']:
            raise ValueError('Downloaded part failed verification: ' + part['name'])
        temporary.replace(path)
        print('Verified:', part['name'], flush=True)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, parts))
    temporary = target.with_suffix(target.suffix + '.assembling')
    with temporary.open('wb') as output:
        for path in paths:
            with path.open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if temporary.stat().st_size != data['archive']['bytes'] or digest(temporary) != data['archive']['sha256']:
        raise ValueError('Assembled archive failed verification')
    temporary.replace(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(download(args.manifest, args.destination))
