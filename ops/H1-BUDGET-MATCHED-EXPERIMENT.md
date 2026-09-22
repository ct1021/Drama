# H1-R1 budget-matched experiment

## Question

At the same total parameter budget and the same 100K Boxing protocol, does a routed temporal core improve the model's action-consequence learning enough to improve policy return?

The architectural hypothesis is that the latent, reward, and continuation variables should not all use a fully shared recurrent transition. H1-R1 assigns block-triangular recurrent connections and partitioned normalization to those roles, allowing shared information where it is useful while preventing private reward/continuation channels from leaking into the latent state.

## Why this comparison is fair

The archived A run is the control: `lightweight`, seed 3710, 100K Boxing interactions, world-model parameters 7,161,603 and total model parameters 10,581,267.

The new candidate is `lightweight-routed-budgetmatched`: H1-R1 routing, `HiddenStateDim=640`, and Actor hidden width 192. The measured counts are world model 7,163,163, agent 3,418,128, total 10,581,291. The total differs from A by 24 parameters (0.00023%). The extra recurrent capacity is paid for by a smaller policy actor, so the test is a parameter reallocation test rather than a free capacity increase.

Existing C and D runs remain structural context only: C isolates restricted readouts at 6,964,995 world-model parameters; D isolates H1-R1 at 5,762,819. Their unequal budgets prevent using them as the primary causal comparison.

## Execution gate

1. Run local profile and routing tests.
2. Build the candidate on the 5090 without training; assert all parameter counts, finite parameters, optimizer coverage, and one short forward/backward/imagination smoke test.
3. Run one 100K candidate from scratch with seed 3710. Do not queue another seed or another architecture automatically.
4. Compare the fixed 10-episode endpoint and the 60K–100K mean against archived A. Also report learning-curve area, evaluation variance, model/agent loss finiteness, runtime, and GPU memory.

## Decision rule

The candidate passes the first gate only if it completes 100K without non-finite metrics or missing checkpoints and improves both the late-window mean and the fixed endpoint over A by at least 10% relative, while staying within 1.25x A wall time. A result below this gate is a falsification or refinement signal, not evidence to add more seeds. A pass justifies a pre-registered three-seed replication and transition-prediction diagnostics; it does not by itself establish generalization or a complete Drama reproduction.

## Reproducibility boundary

The run uses the existing public configuration, seed 3710, `ALE/Boxing-v5`, 100,000 environment interactions, evaluation every 10,000 steps with 10 episodes, and the repository commit recorded in `manifest.json`. Checkpoints contain model and policy weights only; they do not support exact optimizer/replay/RNG continuation.
