import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from powerlifting_performance.config import ID, TARGET
from powerlifting_performance.validation import duplicate_groups

LEVEL_KEYS = ["meet_key", "profile_key"]
KEY_COLS = LEVEL_KEYS + ["context_key"]


def key_stats(keys: pd.Series, level: pd.Series) -> pd.DataFrame:
    """Mean relative level and number of rows for each key value."""
    return level.groupby(keys).agg(["mean", "size"])


def map_key_stats(keys: pd.Series, stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return keys.map(stats["mean"]).to_numpy(), keys.map(stats["size"]).fillna(0).to_numpy()


class LevelLightGBM(RegressorMixin, BaseEstimator):
    """LightGBM on the tree features plus two level features for each key of LEVEL_KEYS.

    The relative level of a train row is its TotalKg divided by the median TotalKg of its context (same sex,
    equipment and 10 kg bodyweight bin). For a row and a key, `<key>_level` is the mean relative level of the
    train rows sharing its key, and `<key>_count` their number. On the training rows they are computed
    out-of-fold, exact duplicates kept together, so that no row sees its own total.

    n_seeds: one LightGBM per seed, predictions averaged.
    """

    def __init__(self, n_seeds: int = 1):
        self.n_seeds = n_seeds

    def fit(self, X: pd.DataFrame, y: pd.Series):
        F = X.drop(columns=KEY_COLS)
        groups = duplicate_groups(F)
        level = y / X.context_key.map(y.groupby(X.context_key).median())

        for key in LEVEL_KEYS:
            mean, count = np.full(len(X), np.nan), np.zeros(len(X))
            for tr, va in GroupKFold(5, shuffle=True, random_state=1).split(F, y, groups):
                mean[va], count[va] = map_key_stats(X[key].iloc[va], key_stats(X[key].iloc[tr], level.iloc[tr]))
            F[f"{key}_level"], F[f"{key}_count"] = mean, count
        self.stats_ = {key: key_stats(X[key], level) for key in LEVEL_KEYS}

        self.models_ = []
        for seed in range(self.n_seeds):
            tr, es = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(F, y, groups))
            model = lgb.LGBMRegressor(
                n_estimators=10000, learning_rate=0.03, num_leaves=63, min_child_samples=40, cat_smooth=10.0,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=seed,
            )
            model.fit(F.iloc[tr], y.iloc[tr], eval_set=[(F.iloc[es], y.iloc[es])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
            self.models_.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        F = X.drop(columns=KEY_COLS)
        for key, stats in self.stats_.items():
            F[f"{key}_level"], F[f"{key}_count"] = map_key_stats(X[key], stats)
        return np.mean([model.predict(F) for model in self.models_], axis=0)


def blend_with_twins(train: pd.DataFrame, test: pd.DataFrame, predictions: np.ndarray,
                     weight: float = 0.8) -> np.ndarray:
    """For the test rows identical to train rows on every feature (often the same performance entered in several
    divisions), predict weight * mean total of these train rows + (1 - weight) * the model's prediction.

    weight = 0.8 was the best on train rows with a twin (RMSE 37 kg, against 66 kg for the model alone).
    """
    features = [c for c in train.columns if c not in [ID, TARGET]]
    twin_total = test[features].merge(train.groupby(features, as_index=False)[TARGET].mean(),
                                      on=features, how="left")[TARGET].to_numpy()
    return np.where(np.isnan(twin_total), predictions, weight * twin_total + (1 - weight) * predictions)
