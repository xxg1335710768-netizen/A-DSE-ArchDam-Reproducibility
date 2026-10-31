# FE surrogate dataset

`SurrogateDataset.xlsx` contains the FE data used in the reported experiments.

- `TrainX`: 400 x 16 development inputs.
- `TrainY`: 400 x 41 development FE responses.
- `ValX`: 80 x 16 independent test inputs.
- `ValY`: 80 x 41 independent test FE responses.
- Missing values: none.
- Dataset SHA-256: `1f9aa8db86de10d3024856d906ef5324724cf1661039ab6b2bfad9bca58ff353`.

The first test input is `[1, ..., 1]`. `Volu` and `St` are the two optimization objectives. The remaining columns are 39 fixed maximum/minimum principal-stress responses. See the repository-level `DATA_DICTIONARY.md` for symbols, units, bounds, and preprocessing.
