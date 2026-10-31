# Ablation result protocol

The public logs preserve the settings actually used when each experiment was run.

| Method | Population x generations | Interpretation |
|---|---:|---|
| Full | 200 x 200 | formal A-DSE result |
| M1 | 100 x 100 | fixed-space online trajectory |
| M2 | 200 x 200 | formal uniform-expansion ablation |
| M3 | 200 x 200 | formal random-AL ablation |
| M4 | 100 x 100 | legacy no-accuracy-trigger trajectory |

`M1/pf_replay_200x200` re-evaluates Pareto search with 200 x 200 on the surrogate/archive states saved from the M1 trajectory. It does not rerun FE validation, active learning, or surrogate retraining and therefore is not equivalent to an online 200 x 200 experiment.

Consequently, Full/M2/M3 form the strictly matched 200 x 200 subset. M1 and M4 remain useful mechanism diagnostics, but their absolute HV and FE-cost values should not be interpreted as a fully controlled search-budget comparison with that subset.
