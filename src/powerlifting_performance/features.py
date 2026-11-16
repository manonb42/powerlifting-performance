import pandas as pd

from powerlifting_performance.config import CONTEXT_KEY, FEATURES, MEET_KEY, PROFILE_KEY, TARGET


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Replace the weight class by the derived features.

    Args:
        X: Raw feature columns.

    Returns:
        X without WeightClassKg, with age_half (age ending in .5), class_limit and is_plus_class (the weight class as
        numbers, "93+" -> 93 and 1) and bw_ratio (bodyweight / class limit).
    """
    class_limit = pd.to_numeric(X.WeightClassKg.str.rstrip("+"), errors="coerce")
    return X.drop(columns="WeightClassKg").assign(
        age_half=(X.Age % 1 == 0.5).astype(int), class_limit=class_limit,
        is_plus_class=X.WeightClassKg.str.endswith("+").astype(int), bw_ratio=X.BodyweightKg / class_limit,
    )


def level_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Build the keys grouping the rows for the level features.

    Args:
        df: Raw data.

    Returns:
        The meet_key, profile_key and context_key string columns.
    """
    df = df.assign(birth_year=(df.MeetYear - df.Age).round().astype(int), bodyweight_bin=df.BodyweightKg // 10)
    keys = {"meet_key": MEET_KEY, "profile_key": PROFILE_KEY, "context_key": CONTEXT_KEY}
    return pd.DataFrame({name: df[cols].astype(str).agg("|".join, axis=1) for name, cols in keys.items()})


def duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Number the groups of rows with identical features.

    Args:
        df: Raw data.

    Returns:
        Group id of each row, shared by exact duplicates.
    """
    return df.groupby(FEATURES, dropna=False).ngroup()


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the model inputs from the raw data.

    Args:
        train: Raw train set.
        test: Raw test set.

    Returns:
        X_train, y_train in kg and X_test: raw columns, derived features, keys and duplicate groups.
    """
    def build(df: pd.DataFrame) -> pd.DataFrame:
        return add_derived_features(df[FEATURES]).join(level_keys(df)).assign(duplicate_group=duplicate_groups(df))

    return build(train), train[TARGET], build(test)
