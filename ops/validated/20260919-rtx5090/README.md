# RTX 5090 environment verification — 2026-09-19

All six checks in `verify.json` passed on the commit in `tested-commit.txt`.
`pip-check.txt` reports no broken dependencies. No optimizer steps or RL training
were run. These results do not validate a research hypothesis or benchmark score.

- Ubuntu 22.04.5, Python 3.12.3, driver 580.76.05, CUDA toolkit 12.8.93.
- RTX 5090 / SM120 / 32607 MiB. Container memory limit: 90 GiB.
- Torch 2.7.1+cu128, Triton 3.3.1; repository-local Mamba Python code.
- Original base environment (Torch 2.8.0+cu128, Triton 3.4.0) was left intact.
- `versions.txt` is the complete observed package version inventory, not a
  standalone cross-platform installer. Use the parent bootstrap/build scripts;
  CUDA packages and the local SM120 build require their designated sources.
- Official wheel origins and hashes: `../../wheels-cu128.json`.
- Custom causal-conv1d wheel: `causal_conv1d-1.5.2+sm120-cp312-cp312-linux_x86_64.whl`.
  SHA256: `20468b74716b8d5b252588b0263629454cedcd053e2bb8e58f53da6730f2a0db`.
  Binary and full installation logs are stored separately, outside Git.

Scope: CUDA operator forward/backward; vendored Mamba forward/backward and
8-frame prefill plus 8 recurrent steps (float32); Drama imports; three Atari
ROM/wrapper checks with 8 NOOP actions each; small Drama imagination call using
default CUDA Graph. Full training, torch.compile, mixed precision, DMC, production
batch sizes, and other GPU generations remain unverified.

The initial d_model=128 smoke model violated the convolution's projection-stride
alignment. The final d_model=256 smoke uses eight heads and an aligned projection.
No Drama model configuration was changed for this test adjustment.
