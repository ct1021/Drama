# First online RL run: Boxing / seed 3710 / 100K

Authorized 2026-09-19. Start one baseline only; hourly observation does not authorize
additional seeds, method variants, instance purchases or automatic relaunches.

## Frozen scope

- Baseline: upstream YAML four-stage encoder and original Mamba2/DFS/AC losses.
  Call this the public-configuration baseline, not paper-exact DramaXS.
- Boxing is the first game in the previously proposed Boxing/Krull subset. This
  run measures baseline feasibility and learning; it cannot establish H1 or a
  general Atari100K result. The architecture candidate remains unimplemented.
- Seed 3710; 100000 training environment actions, repeat=4 except early episode /
  life termination as implemented by the training wrapper. Evaluation interactions
  are separate and never enter replay. No pretrained weights or teacher actions.
- Original world-model batch 16 x 128; imagination 1024 x 16, context 8; reality
  context 16. Default AMP and CUDA Graph retained. `torch.compile` disabled for this
  first frozen profile; record this efficiency difference, do not compare its time
  directly to a compiled method variant.
- Every 10000 actions: exactly ten complete greedy-policy evaluation episodes,
  fixed emulator/action seeds `200000 + 3710 * 10 + [0..9]`; isolated Torch/NumPy/
  Python RNG streams. Uses the upstream evaluation image/frame-skip wrapper.
  Finished evaluation slots are not restarted or counted twice. Separate histories
  per batch row; contexts are recreated at every evaluation call.
- Weights every 2000 actions (latest), final weights at 100000; all scalar metrics
  and per-episode evaluation returns saved locally. No cloud logging or media.
  Weight snapshots are not exact training-resume checkpoints (no replay/optimizer).
- One run has a 12-hour external timeout and an 11.9-hour cooperative limit.
  At the observed rental price RMB 2.88/hour, its 12-hour window is RMB 34.56.
  This bounds the training process, not provider billing: the instance remains on.
  No next job is launched automatically.

## Common execution corrections (apply equally to later comparison groups)

`train.py`: respect caller GPU visibility (single-card default 0); initialize
discrete-action detection within the training function; count actual completed
interactions from 1 through N; use the reset observation on the next transition;
seed training random actions; perform final evaluation; close the training env.
`envs/my_atari.py`: honor the supplied emulator seed on first reset.
No loss, model architecture, batch, context length or DFS sampling change.

`run_atari100k.py` supplies a persistent local scalar logger and fixed-episode
evaluation implementation. It records resolved configuration, exact commit,
parameter counts, run phase and GPU peak memory. The original W&B logger in
disabled mode did not preserve the research metrics we need.

## Launch gate and checks

Run `python ops/test_training_protocol.py` and one separate full-batch integration
run of 1040 interactions plus two complete evaluation episodes. This verifies real
world-model and actor/critic updates and checkpoint creation; it is not a performance
result. Only after that gate passes, run `bash ops/launch_boxing.sh` once and confirm
the 100K process has passed warmup and is writing both model and policy losses.

Hourly checks read status, process identity, log errors, disk/GPU and evaluation
records. Notify on completion, failure, a stalled process or a required decision;
otherwise retain a compact local check record. Do not alter hyperparameters or
start another experiment based on interim scores.
