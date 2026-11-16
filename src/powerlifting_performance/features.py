import pandas as pd

from powerlifting_performance.config import CONTEXT_KEY, FEATURES, MEET_KEY, PROFILE_KEY, TARGET


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Hints of the athlete's level that the raw columns only hold implicitly.

    - age_half: age ending in .5, i.e. approximated from a birth year.
    - class_limit, is_plus_class: the weight class as numbers ("93+" -> 93 and 1). They replace the
      WeightClassKg category, which only adds noise once they are there (78.28 vs 78.40 RMSE).
    - bw_ratio: bodyweight / class limit; close to 1 means the athlete made weight on purpose.

    is_international (Country != MeetCountry) was tried and dropped: no gain, the model already has both columns.
    """
    class_limit = pd.to_numeric(X.WeightClassKg.str.rstrip("+"), errors="coerce")
    return X.drop(columns="WeightClassKg").assign(
        age_half=(X.Age % 1 == 0.5).astype(int), class_limit=class_limit,
        is_plus_class=X.WeightClassKg.str.endswith("+").astype(int), bw_ratio=X.BodyweightKg / class_limit,
    )


def level_keys(df: pd.DataFrame) -> pd.DataFrame:
    """String keys grouping the rows for the level features: meet_key, profile_key and context_key."""
    df = df.assign(birth_year=(df.MeetYear - df.Age).round().astype(int), bodyweight_bin=df.BodyweightKg // 10)
    keys = {"meet_key": MEET_KEY, "profile_key": PROFILE_KEY, "context_key": CONTEXT_KEY}
    return pd.DataFrame({name: df[cols].astype(str).agg("|".join, axis=1) for name, cols in keys.items()})


def duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """One group id per set of rows with identical features.

    Identical rows are mostly the same performance entered in several divisions,
    so they must stay in the same fold.
    """
    return df.groupby(FEATURES, dropna=False).ngroup()


def prepare_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train (in kg) and X_test for LightGBM: raw columns, derived features, then the keys."""
    build = lambda df: (add_derived_features(df[FEATURES]).join(level_keys(df))
                        .assign(duplicate_group=duplicate_groups(df)))
    return build(train), train[TARGET], build(test)
