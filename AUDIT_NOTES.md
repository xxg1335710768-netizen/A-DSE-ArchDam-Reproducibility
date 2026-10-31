# Public-package audit

Audit date: 2026-10-31.

The repository was checked against `Manuscript_R1.docx` and the retained experiment metadata. The following release corrections were made without changing the source FE dataset or original experiment directories:

1. The surrogate comparison was renamed from Table 5 to Table 6 to match the manuscript.
2. The previous `time_min` values, which represented a nine-configuration search workflow, were removed. They were replaced by the independently rerun fixed-configuration fitting times reported in the manuscript.
3. Table 7 PICP and MPIW files were separated from Table 6. Unreported uncertainty-ranking diagnostics were removed from the public main result set.
4. The sample-size workbook was renamed from `Figure11_data.xlsx` to `Figure10_data.xlsx` to match the manuscript, and redundant training diagnostics were omitted.
5. A duplicate optimizer workbook was removed. The retained optimizer files use current fronts rather than cumulative historical fronts.
6. Machine-specific absolute paths were replaced by repository-relative paths.
7. The M1 and M4 ablation logs were labelled accurately: their online searches used 100 x 100 NSGA-II, whereas Full, M2, and M3 used 200 x 200. M1 includes a clearly identified post-hoc 200 x 200 PF replay; it is not presented as a new online run.
8. The package verifier now checks manuscript Table 6/7 values, repeated-run completeness, optimizer budgets, ablation protocol labels, checksums, and path hygiene.
9. The released Type-0 geometry checker was synchronized with the constraint formulation used by the optimizer (`0.35` overhang limit and the executed final-layer branch). Its LHS equivalence test now passes.

The pre-audit state remains available locally on branch `backup/pre-audit-20261031`.
