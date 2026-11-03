# Powerlifting Performance

Predict the total (`TotalKg`) lifted by a powerlifter at a meet, from the OpenPowerlifting-derived data in `data/`.
The metric is RMSE.

## Requirements

- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Python: uv installs the right version automatically, based on `.python-version`

To install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

```bash
git clone https://github.com/manonb42/powerlifting-performance.git
cd powerlifting-performance
uv sync
```

`uv sync` creates a `.venv` virtual environment and installs the exact dependency versions listed in `uv.lock`.

## Usage

Cross-validate the model (LightGBM with level features), fit it on the whole train set and write a submission to
`submissions/submission_lightgbm.csv`:

```bash
uv run python -m powerlifting_performance.train
```

For the test rows identical to train rows on every column, the submission blends the prediction with the totals of
these train rows. The cross-validation score does not include this blend.

## Code

All the code is in `src/powerlifting_performance/`.

| File | Role |
|---|---|
| `features.py` | Features: raw columns, derived features, meet statistics, level keys |
| `model.py` | The model, `LevelLightGBM`, and `blend_with_twins` |
| `train.py` | Command line: cross-validation, fit, submission |
| `validation.py` | Cross-validation that keeps exact duplicates in the same fold |
| `config.py`, `data.py` | Paths, column names, reading the data and writing a submission |
