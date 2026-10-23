import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from powerlifting_performance.config import CAT_COLS
from powerlifting_performance.validation import duplicate_groups


def make_linear_pipeline(bodyweight_spline: bool = True) -> Pipeline:
    """Linear regression on one-hot categories, a spline on age and (optionally) on log bodyweight."""
    return make_pipeline(
        make_column_transformer(
            (OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist"), CAT_COLS),
            (SplineTransformer(n_knots=10), ["Age"]),
            (SplineTransformer(n_knots=10) if bodyweight_spline else "passthrough", ["log_bodyweight"]),
            remainder="passthrough",
        ),
        LinearRegression(),
    )


class TotalKgRegressor(RegressorMixin, BaseEstimator):
    """Predicts TotalKg from a linear model fitted on log(TotalKg).

    by_sex: fit one model for men and one for everyone else (women and Mx).
    correct_bias: exp(log prediction) estimates the median; multiply by exp(residual variance / 2)
    to estimate the mean instead.
    """

    def __init__(self, bodyweight_spline: bool = True, by_sex: bool = True, correct_bias: bool = True):
        self.bodyweight_spline = bodyweight_spline
        self.by_sex = by_sex
        self.correct_bias = correct_bias

    def _split(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        if self.by_sex:
            return {"men": X.Sex == "M", "others": X.Sex != "M"}
        return {"all": pd.Series(True, index=X.index)}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        log_y = np.log(y)
        self.models_, self.log_shifts_ = {}, {}
        for name, rows in self._split(X).items():
            model = make_linear_pipeline(self.bodyweight_spline).fit(X[rows], log_y[rows])
            residuals = log_y[rows] - model.predict(X[rows])
            self.models_[name] = model
            self.log_shifts_[name] = residuals.var() / 2 if self.correct_bias else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_pred = pd.Series(np.nan, index=X.index)
        for name, rows in self._split(X).items():
            if rows.any():
                log_pred[rows] = self.models_[name].predict(X[rows]) + self.log_shifts_[name]
        return np.exp(log_pred.to_numpy())


def make_gradient_boosting(random_state: int = 0) -> HistGradientBoostingRegressor:
    """Gradient boosting on TotalKg in kg, with the pandas categorical columns handled natively.

    Early stopping on 10 % of the training rows picks the number of trees.
    """
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=2000,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=random_state,
    )


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
