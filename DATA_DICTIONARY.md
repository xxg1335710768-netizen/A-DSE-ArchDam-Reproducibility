# Data dictionary

## Input variables

The 16 inputs are dimensionless multipliers of the initial arch-dam geometry. The initial design is therefore `[1, ..., 1]`. The physical parameter equals its initial value multiplied by the corresponding `x` value.

| Column | Parameter | Unit | Initial value | Lower multiplier | Upper multiplier |
|---|---|---:|---:|---:|---:|
| `x1` | `k_c` | - | -0.50 | 0.870000 | 1.130000 |
| `x2` | `k_b` | - | 0.15 | 0.870000 | 1.130000 |
| `x3` | `z_s` | m | 97.80 | 0.935000 | 1.065000 |
| `x4` | `y_s` | m | 29.31 | 0.970000 | 1.080000 |
| `x5` | `T_m0` | m | 51.63 | 0.973452 | 1.060000 |
| `x6` | `T_m1` | m | 44.06 | 0.950634 | 1.062991 |
| `x7` | `T_m2` | m | 34.42 | 0.883000 | 1.015480 |
| `x8` | `T_m3` | m | 10.00 | 1.000000 | 1.079500 |
| `x9` | `T_l0` | m | 52.92 | 0.883000 | 1.117000 |
| `x10` | `T_l1` | m | 47.54 | 0.857000 | 1.143000 |
| `x11` | `T_l2` | m | 35.77 | 0.857000 | 1.143000 |
| `x12` | `T_l3` | m | 14.13 | 0.883000 | 1.117000 |
| `x13` | `T_r0` | m | 52.25 | 0.942085 | 1.117000 |
| `x14` | `T_r1` | m | 52.31 | 0.857000 | 1.085800 |
| `x15` | `T_r2` | m | 37.44 | 0.857000 | 1.143000 |
| `x16` | `T_r3` | m | 14.09 | 0.883000 | 1.117000 |

## Output variables

| Columns | Unit | Meaning |
|---|---:|---|
| `Volu` | 10^4 m^3 | Dam concrete volume |
| `St` | kPa | Structural safety objective used in the manuscript |
| `UP_S1_Pxxx`, `DOWN_S1_Pxxx` | MPa | Maximum principal stress at an upstream/downstream response point |
| `UP_S3_Pxxx`, `DOWN_S3_Pxxx` | MPa | Minimum principal stress at an upstream/downstream response point |

`UP` and `DOWN` identify the upstream and downstream surfaces, and `Pxxx` identifies the fixed response-point index. Negative stress denotes tension and positive stress denotes compression. These 39 columns are principal-stress responses; they are not von Mises equivalent stresses.

The 39 response locations were selected using only the 400 development samples: a response was retained when its development maximum exceeded 9.05 MPa or its development minimum was below -1.35 MPa. The independent test set was not used for response selection.

## Workbook validation

- `TrainX`: 400 rows, 16 columns, no missing values;
- `TrainY`: 400 rows, 41 columns, no missing values;
- `ValX`: 80 rows, 16 columns, no missing values;
- `ValY`: 80 rows, 41 columns, no missing values.

The first `ValX` row is the initial design `[1, ..., 1]`.

## Standardization

For each experimental split, each input feature is standardized using only the parameter-fitting training subset:

```text
x_standardized[d] = (x[d] - training_mean[d]) / training_population_std[d]
```

The same statistics are applied without refitting to the validation, calibration, and independent test sets. Each of the 41 outputs is standardized independently using the same training-only rule. During adaptive retraining, the statistics are refitted after the training archive is updated.

NSGA-II, MOEA/D, and MOPSO-CD operate in the original multiplier space. Network standardization is performed internally by the surrogate.
