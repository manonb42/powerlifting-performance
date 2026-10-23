# Powerlifting Performance


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

Train the model, print the cross-validation scores and write a submission to `submissions/`:

```bash
uv run python -m powerlifting_performance.train
uv run python -m powerlifting_performance.train --model gradient_boosting
uv run python -m powerlifting_performance.train --model linear
```

LightGBM options: `--seeds N` averages N models (default 1), `--twins` blends the submission with the totals of
exact duplicates in train, `--skip-cv` only fits and writes the submission. The best submission so far:

```bash
uv run python -m powerlifting_performance.train --seeds 5 --twins --skip-cv
```
