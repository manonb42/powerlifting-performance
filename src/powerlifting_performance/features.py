import pandas as pd

from powerlifting_performance.config import CAT_COLS, CONTEXT_KEY, MEET_KEY, NUM_COLS, PROFILE_KEY, TARGET


RARE = "Rare"


def category_dtypes(train: pd.DataFrame, min_count: int = 20,
                    max_categories: int | None = 250) -> dict[str, pd.CategoricalDtype]:
    """Categories kept for each column: the most frequent ones, with at least `min_count` rows in train.

    HistGradientBoostingRegressor accepts at most 255 categories per column (Federation has ~400),
    and rare categories are noise anyway: they all go to `RARE`.
    """
    dtypes = {}
    for col in CAT_COLS:
        counts = train[col].value_counts()
        kept = counts[counts >= min_count].index[:max_categories]
        dtypes[col] = pd.CategoricalDtype([*kept, RARE])
    return dtypes


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


def build_tree_features(df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    X = df[NUM_COLS].copy()
    for col, dtype in dtypes.items():
        X[col] = df[col].where(df[col].isin(dtype.categories), RARE).astype(dtype)
    return add_derived_features(X, df)


def level_keys(df: pd.DataFrame) -> pd.DataFrame:
    """String keys grouping the rows for the level features: meet_key, profile_key and context_key."""
    df = df.assign(birth_year=(df.MeetYear - df.Age).round().astype(int), bodyweight_bin=df.BodyweightKg // 10)
    join = lambda cols: df[cols].astype(str).agg("|".join, axis=1)
    return pd.DataFrame({"meet_key": join(MEET_KEY), "profile_key": join(PROFILE_KEY),
                         "context_key": join(CONTEXT_KEY)})


def prepare_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train (in kg) and X_test for LightGBM: the tree features with every category kept
    (LightGBM has no limit on their number), plus the level keys."""
    dtypes = category_dtypes(train, min_count=1, max_categories=None)
    build = lambda df: build_tree_features(df, dtypes).join(level_keys(df))
    return build(train), train[TARGET], build(test)
