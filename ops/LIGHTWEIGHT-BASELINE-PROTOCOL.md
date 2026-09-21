# First lightweight Boxing 100K baseline

2026-09-21: user explicitly requested proceeding to initial training on the cloned instance.
Start exactly one lightweight baseline after a separate 1200-interaction integration check.
No candidate architecture, additional seeds or automatic experiment queue is authorized here.

- Game Boxing, seed3710, from scratch, exactly100000 training interactions.
- Profile `lightweight`: AE Encoder/Decoder Mults `[1,2,3,4,4]` only.
  World model7,161,603 parameters; actor/critic3,419,664 parameters.
- Original Mamba2 (head_dim64), losses, DFS, batches16×128 and imagination1024×16,
  context8/reality16, optimizers and policy interface unchanged.
- AMP/CUDA Graph enabled; compile disabled. This is a lightweight public-code baseline,
  not paper-exact DramaXS or an H1 result.
- Every10K: fixed10 complete greedy-policy evaluation episodes, same protocol as the prior run.
- New run directory `boxing-lightweight-baseline-s3710-100k-20260921`.
  Preserve prior public-config run and its complete archive.
- Planned training time about5–6hours, provisional. External timeout8hours, cooperative7.9hours.
  Timeouts preserve interrupted weights but do not claim100K completion or turn off provider billing.
- Weights only: no replay/optimizer/RNG snapshot, so no exact training resume.
- Integration gate: profile unit tests, existing interaction/reset/evaluation tests, then1200
  real interactions with2 complete evaluation episodes, finite WM/AC losses and final weights.
  Integration is software validation, not candidate performance evidence.

The four-group architecture design remains a separate proposed experiment, not a queue.
