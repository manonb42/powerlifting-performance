import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, train_test_split

from powerlifting_performance.config import KEY_COLS, LEVEL_KEYS, TREE_CAT_COLS

PARAMS = dict(learning_rate=0.01, num_leaves=53, min_child_samples=50,
              subsample=0.9518264600084477, subsample_freq=1, colsample_bytree=0.4501644935675392,
              reg_lambda=0.029083257971357097, cat_smooth=24.981310172342987)
RARE = "Rare"


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
    """LightGBM on the features of `features.prepare_features` plus the level features (`fit_level_features`), one model
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


class TargetLightGBM(RegressorMixin, BaseEstimator):
    """LevelLightGBM with one seed, fitted on another target: "relative" (TotalKg divided by the median TotalKg of its
    context) or "log" (log of TotalKg, predictions taken back with exp)."""

    def __init__(self, target: str = "log"):
        self.target = target

    def reference(self, X: pd.DataFrame) -> np.ndarray:
        return X.context_key.map(self.reference_).fillna(self.reference_.median()).to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.target == "relative":
            self.reference_ = y.groupby(X.context_key).median()
            y = y / self.reference(X)
        else:
            y = np.log(y)
        self.model_ = LevelLightGBM(seeds=(0,)).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prediction = self.model_.predict(X)
        return prediction * self.reference(X) if self.target == "relative" else np.exp(prediction)


class GradientBoostingExpert(RegressorMixin, BaseEstimator):
    """scikit-learn gradient boosting on the raw and derived columns. It accepts at most 255 categories per column: each
    keeps its 250 most frequent categories with at least 20 training rows, the others are grouped in RARE."""

    @staticmethod
    def frequent_categories(values: pd.Series) -> list:
        """The 250 most frequent values with at least 20 rows."""
        counts = values.value_counts()
        return list(counts[counts >= 20].index[:250])

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        F = X.drop(columns=KEY_COLS)
        for col, kept in self.categories_.items():
            F[col] = F[col].where(F[col].isin(kept), RARE).astype(pd.CategoricalDtype([*kept, RARE]))
        return F

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.categories_ = {col: self.frequent_categories(X[col]) for col in TREE_CAT_COLS}
        self.model_ = HistGradientBoostingRegressor(
            categorical_features="from_dtype", learning_rate=0.05, max_iter=2000, max_leaf_nodes=63,
            min_samples_leaf=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=0,
        )
        self.model_.fit(self.features(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.features(X))


class CatBoostExpert(RegressorMixin, BaseEstimator):
    """CatBoost on the raw and derived columns, plus the level features of LevelLightGBM if `level`. It stops early on a
    random 10 % of the training rows."""

    def __init__(self, level: bool = True):
        self.level = level

    def features(self, X: pd.DataFrame, columns: dict[str, np.ndarray]) -> pd.DataFrame:
        return X.drop(columns=KEY_COLS).astype(dict.fromkeys(TREE_CAT_COLS, object)).assign(**columns)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        columns, self.stats_ = fit_level_features(X, y) if self.level else ({}, {})
        F_tr, F_es, y_tr, y_es = train_test_split(self.features(X, columns), y, test_size=0.1, random_state=0)
        self.model_ = CatBoostRegressor(iterations=3000, learning_rate=0.08, depth=6, od_type="Iter", od_wait=100,
                                        random_seed=0, verbose=0, cat_features=TREE_CAT_COLS)
        self.model_.fit(F_tr, y_tr, eval_set=(F_es, y_es), use_best_model=True)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.features(X, level_features(X, self.stats_)))
