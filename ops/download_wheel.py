"""Download a large official wheel using HTTP ranges; require its published SHA256."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
import urllib.request
from urllib.parse import urlparse, unquote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("sha256")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if len(args.sha256) != 64 or any(c not in "0123456789abcdef" for c in args.sha256):
        raise ValueError("Expected the SHA256 published by the official index")
    if urlparse(args.url).hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}:
        raise ValueError("Only official PyTorch download hosts are accepted")
    filename = unquote(Path(urlparse(args.url).path).name)
    if not filename.endswith(".whl"):
        raise ValueError("Expected a wheel URL")
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / filename

    def digest(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    if target.exists():
        if digest(target) != args.sha256:
            raise ValueError("Existing wheel has an unexpected checksum")
        print("Already verified:", filename, flush=True)
        return
    request = urllib.request.Request(args.url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise ValueError("Server did not support HTTP ranges")
        size = int(response.headers["Content-Range"].split("/")[-1])
    count = 16
    step = (size + count - 1) // count
    print(f"Downloading {filename}: {size} bytes, {count} connections", flush=True)
    with tempfile.TemporaryDirectory(prefix="wheel-parts-", dir=args.directory) as tmp:
        command = ["curl", "--parallel", "--parallel-max", str(count)]
        parts = []
        for start in range(0, size, step):
            end = min(size - 1, start + step - 1)
            part = Path(tmp) / str(start)
            parts.append((part, end-start+1))
            if len(parts) > 1:
                command.append("--next")
            command.extend(["--fail", "--location", "--silent", "--show-error",
                            "--connect-timeout", "15", "--max-time", "600",
                            "--retry", "3", "--retry-delay", "1",
                            "--range", f"{start}-{end}", "--output", str(part), args.url])
        subprocess.run(command, check=True)
        assembled = Path(tmp) / "assembled.whl"
        with assembled.open("wb") as output:
            for part, expected in parts:
                if part.stat().st_size != expected:
                    raise ValueError("Unexpected range size")
                with part.open("rb") as source:
                    while chunk := source.read(2**20):
                        output.write(chunk)
        if digest(assembled) != args.sha256:
            raise ValueError("Wheel did not match the official index SHA256")
        assembled.replace(target)
    print("SHA256 verified:", filename, flush=True)


if __name__ == "__main__":
    main()
