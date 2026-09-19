"""Verify and extract a portable experiment without installing or training."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def restore(archive, checksum, destination):
    if sha256(archive) != checksum.lower():
        raise ValueError('Archive SHA256 mismatch; nothing extracted')
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination must be new or empty')
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        manifest_member = next(m for m in members if m.name == 'MANIFEST.json')
        if not manifest_member.isfile():
            raise ValueError('Manifest is not a regular file')
        manifest = json.load(tar.extractfile(manifest_member))
        records = manifest['files']
        expected = {r['path']: r for r in records}
        if len(expected) != len(records):
            raise ValueError('Duplicate manifest paths')
        seen = set()
        # Validate all members before writing any archive content.
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or '\\' in member.name or ':' in member.name
                    or member.name in seen):
                raise ValueError('Unsafe/duplicate archive path: ' + member.name)
            seen.add(member.name)
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Path escapes destination')
        if seen != set(expected) | {'MANIFEST.json'}:
            raise ValueError('Archive/manifest file list mismatch')
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as source, target.open('xb') as output:
                while block := source.read(1024 * 1024):
                    h.update(block)
                    output.write(block)
            if member.name != 'MANIFEST.json':
                record = expected[member.name]
                if target.stat().st_size != record['bytes'] or h.hexdigest() != record['sha256']:
                    raise ValueError('Extracted file mismatch: ' + member.name)
    result = dict(archive_sha256=checksum.lower(), verified_files=len(records),
                  training_commit=manifest['training_commit'],
                  exact_training_resume=manifest['exact_training_resume'],
                  destination=str(destination), installed_packages=False, started_training=False)
    (destination / 'RESTORE-VERIFIED.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.sha256, args.destination), indent=2))


if __name__ == '__main__':
    main()
