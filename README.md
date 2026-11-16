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

One command trains the final model and writes the submission to hand in, `submissions/submission_ensemble.csv`:

```bash
uv run python -m powerlifting_performance.train
```

The final model is a stacking of six experts: each is cross-validated on the same folds, and a linear regression with
positive weights combines their out-of-fold predictions. The command prints the RMSE of each expert (the first one is
the main LightGBM alone), the cross-validated RMSE of the stacking and its weights. For the test rows identical to a
train row on every column, the submission is blended with the totals of these train rows; the cross-validation scores
do not include this blend. It takes about three hours.

## Code

All the code is in `src/powerlifting_performance/`.

| File | Role |
|---|---|
| `features.py` | Features: raw columns, derived features, level keys, groups of exact duplicates |
| `model.py` | The main model, `LevelLightGBM` (parameters in `PARAMS`), and the level features |
| `experts.py` | The other experts of the stacking: LightGBM on another target, scikit-learn gradient boosting, CatBoost |
| `train.py` | Command line: experts, stacking, blend with the exact duplicates from train, submission |
| `validation.py` | Cross-validation on the groups of exact duplicates |
| `config.py`, `data.py` | Paths, column lists, reading the data and writing a submission |
