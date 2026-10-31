# Tables 6 and 7

- `table6_by_seed.csv`: R2, RMSE, NLL, and fitting time for seeds 2025--2029.
- `table6_summary.csv`: mean and sample standard deviation across the five seeds.
- `table7_by_seed.csv` and `table7_summary.csv`: 95% PICP and MPIW.
- `fitting_time_by_seed.csv`: timing records for the fixed model configurations.

R2 and RMSE are macro-averages across 41 responses. NLL is evaluated on standardized responses after variance calibration. XGB does not provide predictive variance and therefore has no NLL or interval metrics.
