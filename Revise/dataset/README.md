# FE surrogate dataset

`SurrogateDataset.xlsx` is an unchanged copy of the dataset used by the formal experiments.

- `TrainX`: 400 x 16 development inputs.
- `TrainY`: 400 x 41 development FE responses.
- `ValX`: 80 x 16 independent test inputs.
- `ValY`: 80 x 41 independent test FE responses.
- Missing values: none.
- Dataset SHA-256: `d03a5af15e3ac03dd018f6922578473a206eadfb09be40621104a944e4a35c41`.

The first test input is `[1, ..., 1]`. `Volu` and `St` are the two optimization objectives. The remaining columns are 39 fixed maximum/minimum principal-stress responses. See the repository-level `DATA_DICTIONARY.md` for symbols, units, bounds, and preprocessing.
