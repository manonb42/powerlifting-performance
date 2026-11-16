from pathlib import Path

import pandas as pd

from powerlifting_performance.config import DATA_DIR, ID, SUBMISSIONS_DIR, TARGET


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw train and test sets."""
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def save_submission(ids: pd.Series, predictions, name: str) -> Path:
    """Write predictions in the sample_submission.csv format."""
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / f"submission_{name}.csv"
    pd.DataFrame({ID: ids, TARGET: predictions}).to_csv(path, index=False)
    return path
