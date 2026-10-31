# Response to Comment 2

**Comment 2:** There are no benchmark cases with known solutions to compare against. The FEA data is not made available.

**Response:** The FE dataset and the code used for the reported surrogate and optimization experiments have been released at [repository URL]. The dataset contains 400 development samples and 80 independent test samples, with 16 design variables and 41 FE responses. The repository also includes the data-partition protocol, surrogate implementations, five-run model-comparison results, optimizer comparison, A-DSE ablations, and the FE validation of the final Pareto front.

The 80 test samples were not used for fitting, early stopping, model selection, variance calibration, or response-point selection. A closed-form benchmark is not available for this constrained arch-dam problem; the released test data and Pareto-front FE results provide independent numerical references for checking the reported predictions. Rhino/ANSYS automation and machine-specific solver files are excluded because they are not required to reproduce the experiments from the released FE input-output data.
