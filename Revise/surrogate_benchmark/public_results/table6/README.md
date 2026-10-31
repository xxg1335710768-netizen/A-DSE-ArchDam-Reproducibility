# Tables 6 and 7

`table6_by_seed.csv` and `table6_summary.csv` contain prediction metrics and fixed-configuration fitting times for seeds 2025--2029. R2 and RMSE are macro-averages across 41 physical response columns. Gaussian NLL is evaluated on the standardized response scale after calibration. XGB has no NLL because the released baseline does not produce a predictive variance.

`table7_by_seed.csv` and `table7_summary.csv` contain 95% prediction-interval coverage (PICP) and standardized mean prediction-interval width (MPIW).

`fitting_time_by_seed.csv`, `fitting_time_summary.csv`, and `FITTING_TIME_PROTOCOL.md` retain the timing evidence. These values measure final fitting after hyperparameters were frozen; they are not grid-search times.
