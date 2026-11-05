import pandas as pd

from powerlifting_performance.config import CAT_COLS, CONTEXT_KEY, MEET_KEY, NUM_COLS, PROFILE_KEY, TARGET
from powerlifting_performance.validation import duplicate_groups


def add_derived_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Hints of the athlete's level that the raw columns only hold implicitly.

    - age_half: age ending in .5, i.e. approximated from a birth year.
    - class_limit, is_plus_class: the weight class as numbers ("93+" -> 93 and 1). They replace the
      WeightClassKg category, which only adds noise once they are there (78.28 vs 78.40 RMSE).
    - bw_ratio: bodyweight / class limit; close to 1 means the athlete made weight on purpose.

    is_international (Country != MeetCountry) was tried and dropped: no gain, the model already has both columns.
    """
    X = X.drop(columns="WeightClassKg")
    X["age_half"] = (df.Age % 1 == 0.5).astype(int)
    X["class_limit"] = pd.to_numeric(df.WeightClassKg.str.rstrip("+"), errors="coerce")
    X["is_plus_class"] = df.WeightClassKg.str.endswith("+").astype(int)
    X["bw_ratio"] = df.BodyweightKg / X.class_limit
    return X


def level_keys(df: pd.DataFrame) -> pd.DataFrame:
    """String keys grouping the rows for the level features: meet_key, profile_key and context_key."""
    df = df.assign(birth_year=(df.MeetYear - df.Age).round().astype(int), bodyweight_bin=df.BodyweightKg // 10)
    join = lambda cols: df[cols].astype(str).agg("|".join, axis=1)
    return pd.DataFrame({"meet_key": join(MEET_KEY), "profile_key": join(PROFILE_KEY),
                         "context_key": join(CONTEXT_KEY)})


def prepare_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train (in kg) and X_test for LightGBM: raw columns, derived features, then the keys."""
    build = lambda df: (add_derived_features(df[NUM_COLS + CAT_COLS], df).join(level_keys(df))
                        .assign(duplicate_group=duplicate_groups(df)))
    return build(train), train[TARGET], build(test)
