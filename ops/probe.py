"""Read-only environment inventory. Run with the candidate server Python."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    packages = {}
    for name in ("torch", "torchvision", "torchaudio", "triton", "mamba-ssm",
                 "causal-conv1d", "torchao", "torchtune", "gymnasium", "ale-py",
                 "numpy", "transformers", "wandb"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    disks = {}
    for path in ("/", "/root/autodl-tmp", "/tmp", "/dev/shm"):
        if os.path.exists(path):
            usage = shutil.disk_usage(path)
            disks[path] = {"total_GiB": round(usage.total / 2**30, 2),
                           "free_GiB": round(usage.free / 2**30, 2)}
    limits = {}
    for name in ("memory.max", "cpu.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            limits[name] = path.read_text().strip()
    result = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "container_limits": limits,
        "packages": packages, "disks": disks,
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"]),
        "nvcc": command(["nvcc", "--version"]),
        "compiler": command(["g++", "--version"]),
        "torch_probe": command([sys.executable, "-c",
            "import json,torch; print(json.dumps(dict(version=torch.__version__,"
            "cuda=torch.version.cuda,available=torch.cuda.is_available(),"
            "arch_list=torch.cuda.get_arch_list())))"]),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
