import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, train_test_split

from powerlifting_performance.config import FEATURES, KEY_COLS, LEVEL_KEYS, TARGET, TREE_CAT_COLS

PARAMS = dict(learning_rate=0.01, num_leaves=53, min_child_samples=50,
              subsample=0.9518264600084477, subsample_freq=1, colsample_bytree=0.4501644935675392,
              reg_lambda=0.029083257971357097, cat_smooth=24.981310172342987)


def fit_level_features(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, np.ndarray], dict[str, pd.Series]]:
    """Level features of the training rows, computed out-of-fold, and the statistics to compute them for new rows.

    The relative level of a train row is its TotalKg divided by the median TotalKg of its context (same sex,
    equipment and 10 kg bodyweight bin). For a row and a key of LEVEL_KEYS, `<key>_level` is the mean relative level
    of the train rows sharing its key. On the training rows it is computed out-of-fold, exact duplicates kept together,
    so that no row sees its own total.
    """
    level = y / X.context_key.map(y.groupby(X.context_key).median())
    columns = {}
    for key in LEVEL_KEYS:
        mean = np.full(len(X), np.nan)
        for tr, va in GroupKFold(5, shuffle=True, random_state=1).split(X, y, X.duplicate_group):
            mean[va] = X[key].iloc[va].map(level.iloc[tr].groupby(X[key].iloc[tr]).mean()).to_numpy()
        columns[f"{key}_level"] = mean
    return columns, {key: level.groupby(X[key]).mean() for key in LEVEL_KEYS}


def level_features(X: pd.DataFrame, stats: dict[str, pd.Series]) -> dict[str, np.ndarray]:
    """Level features of new rows, from the statistics of `fit_level_features`."""
    return {f"{key}_level": X[key].map(means).to_numpy() for key, means in stats.items()}


class LevelLightGBM(RegressorMixin, BaseEstimator):
    """LightGBM on the features of `features.prepare_lightgbm` plus the level features (`fit_level_features`), one model
    per seed, predictions averaged.

    The categories are those of the training rows. Each model first stops early on a random 10 % of the training
    rows, which gives its number of trees; it is then refitted on all the training rows with 10 % more trees.
    `params` overrides LightGBM parameters of PARAMS.
    """

    def __init__(self, seeds: tuple[int, ...] = tuple(range(10)), params: dict | None = None):
        self.seeds = seeds
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        columns, self.stats_ = fit_level_features(X, y)
        F = X.drop(columns=KEY_COLS).astype(dict.fromkeys(TREE_CAT_COLS, "category")).assign(**columns)
        self.dtypes_ = F.dtypes.to_dict()

        params = {**PARAMS, **(self.params or {})}
        self.models_ = []
        for seed in self.seeds:
            F_tr, F_es, y_tr, y_es = train_test_split(F, y, test_size=0.1, random_state=seed)
            model = lgb.LGBMRegressor(n_estimators=100000, verbose=-1, random_state=seed, **params)
            model.fit(F_tr, y_tr, eval_X=F_es, eval_y=y_es, callbacks=[lgb.early_stopping(200, verbose=False)])
            n_trees = int(1.1 * model.best_iteration_)
            model = lgb.LGBMRegressor(n_estimators=n_trees, verbose=-1, random_state=seed, **params)
            self.models_.append(model.fit(F, y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        F = X.drop(columns=KEY_COLS).assign(**level_features(X, self.stats_)).astype(self.dtypes_)
        return np.mean([model.predict(F) for model in self.models_], axis=0)


def blend_with_twins(train: pd.DataFrame, test: pd.DataFrame, predictions: np.ndarray,
                     weight: float = 0.8) -> np.ndarray:
    """For the test rows identical to train rows on every feature (often the same performance entered in several
    divisions), predict weight * mean total of these train rows + (1 - weight) * the model's prediction.

    weight = 0.8 was the best on train rows with a twin (RMSE 37 kg, against 66 kg for the model alone).
    This only helps when the test set was cut from the same data as train: new athletes have no twin.
    """
    twin_totals = train.groupby(FEATURES, as_index=False)[TARGET].mean()
    twin_total = test[FEATURES].merge(twin_totals, on=FEATURES, how="left")[TARGET].to_numpy()
    return np.where(np.isnan(twin_total), predictions, weight * twin_total + (1 - weight) * predictions)
