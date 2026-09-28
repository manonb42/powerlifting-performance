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
