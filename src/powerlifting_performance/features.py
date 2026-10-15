import numpy as np
import pandas as pd

from powerlifting_performance.config import CAT_COLS, TARGET


def is_wrong_age(df: pd.DataFrame) -> pd.Series:
    """Ages that cannot be right: under 5, or under 8 with an adult bodyweight."""
    return (df.Age < 5) | ((df.Age < 8) & (df.BodyweightKg > 50))


def fix_wrong_ages(df: pd.DataFrame, fill_age: float) -> pd.DataFrame:
    """Flag wrong ages in `age_unknown` and replace them with `fill_age`.

    The fill value does not matter for a linear model: every flagged row gets the same age,
    and the `age_unknown` coefficient absorbs its effect.
    """
    df = df.copy()
    wrong = is_wrong_age(df)
    df["age_unknown"] = wrong.astype(int)
    df.loc[wrong, "Age"] = fill_age
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CAT_COLS + ["Age", "MeetYear", "age_unknown"]].copy()
    X["log_bodyweight"] = np.log(df.BodyweightKg)
    return X


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train (in kg) and X_test."""
    fill_age = train.Age[~is_wrong_age(train)].median()
    train = fix_wrong_ages(train, fill_age)
    test = fix_wrong_ages(test, fill_age)
    return build_features(train), train[TARGET], build_features(test)
