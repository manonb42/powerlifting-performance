import numpy as np
import pandas as pd

from powerlifting_performance.config import ALL_CAT_COLS, CAT_COLS, NUM_COLS, TARGET


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


RARE = "Rare"


def category_dtypes(train: pd.DataFrame, min_count: int = 20,
                    max_categories: int | None = 250) -> dict[str, pd.CategoricalDtype]:
    """Categories kept for each column: the most frequent ones, with at least `min_count` rows in train.

    HistGradientBoostingRegressor accepts at most 255 categories per column (Federation has ~400),
    and rare categories are noise anyway: they all go to `RARE`.
    """
    dtypes = {}
    for col in ALL_CAT_COLS:
        counts = train[col].value_counts()
        kept = counts[counts >= min_count].index[:max_categories]
        dtypes[col] = pd.CategoricalDtype([*kept, RARE])
    return dtypes


def build_tree_features(df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    X = df[NUM_COLS].copy()
    for col, dtype in dtypes.items():
        X[col] = df[col].where(df[col].isin(dtype.categories), RARE).astype(dtype)
    return X


def prepare_tree(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train (in kg) and X_test for the gradient boosting: every raw column, categories as-is."""
    dtypes = category_dtypes(train)
    return build_tree_features(train, dtypes), train[TARGET], build_tree_features(test, dtypes)
