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


def meet_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, statistics of its meet (rows sharing its meet_key) and the size of its profile.

    They describe the level of a meet without its totals: a big international meet full of equipped adult lifters
    is not a small local meet for teenagers. Computed on train + test together: they use no target.
    """
    keys = level_keys(df)
    meet_mean = lambda values: values.groupby(keys.meet_key).transform("mean")
    return pd.DataFrame({
        "meet_n": keys.groupby("meet_key").meet_key.transform("size"),
        "meet_share_raw": meet_mean(df.Equipment == "Raw"),
        "meet_share_multi": meet_mean(df.Equipment == "Multi-ply"),
        "meet_share_men": meet_mean(df.Sex == "M"),
        "meet_mean_age": meet_mean(df.Age),
        "meet_share_youth": meet_mean(df.Age < 18),
        "meet_mean_bw": meet_mean(df.BodyweightKg),
        "meet_share_foreign": meet_mean(df.Country != df.MeetCountry),
        "profile_n": keys.groupby("profile_key").profile_key.transform("size"),
    })


def prepare_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train (in kg) and X_test for LightGBM: raw columns, derived features, meet statistics,
    then the keys."""
    both = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    X = add_derived_features(both[NUM_COLS + CAT_COLS], both).join(meet_stats(both)).join(level_keys(both))
    X_train = X.iloc[:len(train)].assign(duplicate_group=duplicate_groups(train))
    X_test = X.iloc[len(train):].reset_index(drop=True).assign(duplicate_group=duplicate_groups(test))
    return X_train, train[TARGET], X_test
