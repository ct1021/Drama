#!/usr/bin/env bash
# Build the unchanged 1.5.2 kernels for RTX 5090 (SM 12.0).
set -Eeuo pipefail
DATA_DIR="${DRAMA_DATA_DIR:-/root/autodl-tmp/drama-research-20260919}"
ENV_DIR="${DRAMA_ENV_DIR:-$DATA_DIR/envs/drama-cu128}"
PYTHON="$ENV_DIR/bin/python"
SOURCE_DIR="$DATA_DIR/build-sources"
BUILD_WHEELS="$DATA_DIR/built-wheels"
WHEEL="$BUILD_WHEELS/causal_conv1d-1.5.2+sm120-cp312-cp312-linux_x86_64.whl"
mkdir -p "$SOURCE_DIR" "$BUILD_WHEELS"
if [[ ! -f "$WHEEL" ]]; then
    curl --fail --location --connect-timeout 15 --max-time 120 \
        https://pypi.tuna.tsinghua.edu.cn/packages/03/e5/2d2b2e067234c0022ff491ff8e574ca0c67094b2deb61249a2be21789cbb/causal_conv1d-1.5.2.tar.gz \
        -o "$SOURCE_DIR/causal_conv1d-1.5.2.tar.gz"
    echo "9b7d8ec8d07e3590a1dfa010e4e87d1442635c3f96d665a3c1ce3025d8cc4b84  $SOURCE_DIR/causal_conv1d-1.5.2.tar.gz" | sha256sum -c -
    tar -xzf "$SOURCE_DIR/causal_conv1d-1.5.2.tar.gz" -C "$SOURCE_DIR"
    "$PYTHON" - "$SOURCE_DIR/causal_conv1d-1.5.2" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
setup = root / 'setup.py'
text = setup.read_text()
start = text.index('        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_53,code=sm_53")')
end = text.index('\n    # HACK:', start)
text = text[:start] + '''        if bare_metal_version < Version("12.8"):
            raise RuntimeError("SM120 build requires CUDA toolkit >= 12.8")
        cc_flag = ["-gencode", "arch=compute_120,code=sm_120",
                   "-gencode", "arch=compute_120,code=compute_120"]
''' + text[end:]
setup.write_text(text)
version_file = root / 'causal_conv1d' / '__init__.py'
text = version_file.read_text()
assert '__version__ = "1.5.2"' in text
version_file.write_text(text.replace('__version__ = "1.5.2"', '__version__ = "1.5.2+sm120"'))
PY
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    export MAX_JOBS="${MAX_JOBS:-4}"
    "$PYTHON" -m pip wheel --no-deps --no-build-isolation "$SOURCE_DIR/causal_conv1d-1.5.2" -w "$BUILD_WHEELS"
    sha256sum "$WHEEL" > "$BUILD_WHEELS/causal-sm120.sha256"
fi
"$PYTHON" -m pip install --no-deps "$WHEEL"
