from pathlib import Path

import pandas as pd

from powerlifting_performance.config import DATA_DIR, ID, SUBMISSIONS_DIR, TARGET


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw train and test sets."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def save_submission(ids: pd.Series, predictions, name: str, submissions_dir: Path = SUBMISSIONS_DIR) -> Path:
    """Write predictions in the sample_submission.csv format."""
    submissions_dir.mkdir(exist_ok=True)
    path = submissions_dir / f"submission_{name}.csv"
    pd.DataFrame({ID: ids, TARGET: predictions}).to_csv(path, index=False)
    return path
