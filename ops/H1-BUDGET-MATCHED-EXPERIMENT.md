# H1-R1 budget-matched experiment

## Question

At the same total parameter budget and the same 100K Boxing protocol, does a routed temporal core improve the model's action-consequence learning enough to improve policy return?

The architectural hypothesis is that the latent, reward, and continuation variables should not all use a fully shared recurrent transition. H1-R1 assigns block-triangular recurrent connections and partitioned normalization to those roles, allowing shared information where it is useful while preventing private reward/continuation channels from leaking into the latent state.

## What the comparison controls

The archived A run is the control: `lightweight`, seed 3710, 100K Boxing interactions, world-model parameters 7,161,603 and total model parameters 10,581,267.

The new candidate is `lightweight-routed-budgetmatched`: H1-R1 routing, `HiddenStateDim=640`, and Actor hidden width 192. The measured counts are world model 7,163,163, agent 3,418,128, total 10,581,291. The total differs from A by 24 parameters (0.00023%). The extra recurrent capacity is paid for by a smaller policy actor, so the test is a parameter reallocation test rather than a free capacity increase.

This is a combined architecture-and-capacity allocation comparison. It does not isolate routing: recurrent width and actor width also change. Parameter matching does not match FLOPs, cache memory, or training time. No conclusion about action-order generalization follows from Boxing return alone.

Existing C and D runs remain structural context only: C isolates restricted readouts at 6,964,995 world-model parameters; D isolates H1-R1 at 5,762,819. Their unequal budgets prevent using them as the primary causal comparison.

## Execution gate

1. Run local profile and routing tests.
2. Build the candidate on the 5090 without training; assert all parameter counts, finite parameters, optimizer coverage, and one short forward/backward/imagination smoke test.
3. Run one 100K candidate from scratch with seed 3710. Do not queue another seed or another architecture automatically.
4. Compare the fixed 10-episode endpoint and the 60K–100K mean against archived A. Also report learning-curve area, evaluation variance, model/agent loss finiteness, runtime, and GPU memory.

## Decision rule

The candidate passes the first gate only if it completes 100K without non-finite metrics or missing checkpoints and improves both the late-window mean and the fixed endpoint over A by at least 10% relative, while staying within 1.25x A wall time. A result below this gate is a falsification or refinement signal, not evidence to add more seeds. A pass justifies a pre-registered three-seed replication and transition-prediction diagnostics; it does not by itself establish generalization or a complete Drama reproduction.

Operational thresholds (not statistical significance): endpoint >= 67.10, late-window mean >= 53.922, wall time <= 6.089 hours using A's archived 4.871 hours. B's late-window 63.46 and endpoint 64.0 remain the optimization-control reference and must also be reported. Ten evaluation episodes do not replace independent training seeds. Missing a threshold deprioritizes this particular combined configuration; it cannot by itself falsify H1.

The present execution scope is one bounded short training gate followed by this one 100K run. The earlier proposed route ablation is deferred until a same-width, same-budget intervention is specified; simply using C at width 640 would increase world-model parameters to 9,036,443 and introduce another capacity confound. No extra run or seed is queued.

Estimated cost: short training <= 15 minutes; full run approximately 5-6.5 GPU hours, hard process cap 7 hours. No instance shutdown, rental, dependency installation, or additional run is implied. Preserve configuration, code commit, gate records, full metrics/evaluations, logs, and final paired weights before releasing the instance.

## Reproducibility boundary

The run uses the existing public configuration, seed 3710, `ALE/Boxing-v5`, 100,000 environment interactions, evaluation every 10,000 steps with 10 episodes, and the repository commit recorded in `manifest.json`. Checkpoints contain model and policy weights only; they do not support exact optimizer/replay/RNG continuation.
