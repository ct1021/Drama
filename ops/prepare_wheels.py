"""Check published wheel hashes and normalize filenames to embedded metadata."""
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

wheel_dir = Path(sys.argv[1])
manifest = json.loads((Path(__file__).with_name("wheels-cu128.json")).read_text())
for asset in manifest["assets"]:
    source = wheel_dir / asset["filename"]
    if not source.exists():
        continue
    with source.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != asset["sha256"]:
        raise ValueError(f"Checksum mismatch: {source.name}")
    with zipfile.ZipFile(source) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError(f"Expected one wheel METADATA: {source.name}")
        metadata = BytesParser().parsebytes(wheel.read(names[0]))
    tags = source.name[:-4].rsplit("-", 3)[1:]
    canonical_name = "-".join([metadata["Name"].replace("-", "_"), metadata["Version"], *tags]) + ".whl"
    target = wheel_dir / canonical_name
    if target != source and not target.exists():
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    if target != source:
        with target.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != asset["sha256"]:
                raise ValueError(f"Canonical wheel mismatch: {target.name}")
    print(f"Verified {source.name} -> {target.name}")
