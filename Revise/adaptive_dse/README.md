# A-DSE implementation

This directory contains the surrogate-assisted Pareto search, uncertainty-based sampling, GradientSHAP importance calculation, design-space update, and ablation definitions.

The five ablation configurations are defined in `ablation_settings.py` and summarized in `ablation_protocol.csv`. They share a population size of 200, 200 generations per Pareto search, 80 initial FE samples, 30 outer iterations, an active-learning batch size of three, and seed 2025.

Project-specific Rhino/ANSYS automation is not included. Consequently, the full FE-coupled workflow requires the user's own geometry and solver interface. The public code covers the method logic and the FE-independent optimizer comparison.

Full run logs, model checkpoints, Pareto archives, and manuscript plotting workbooks are intentionally omitted.
