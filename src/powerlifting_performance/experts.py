import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

from powerlifting_performance.config import CAT_COLS
from powerlifting_performance.model import KEY_COLS, LevelLightGBM

TREE_CAT_COLS = [c for c in CAT_COLS if c != "WeightClassKg"]
RARE = "Rare"


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

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        F = X.drop(columns=KEY_COLS)
        for col, kept in self.categories_.items():
            F[col] = F[col].where(F[col].isin(kept), RARE).astype(pd.CategoricalDtype([*kept, RARE]))
        return F

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.categories_ = {}
        for col in TREE_CAT_COLS:
            counts = X[col].value_counts()
            self.categories_[col] = list(counts[counts >= 20].index[:250])
        self.model_ = HistGradientBoostingRegressor(
            categorical_features="from_dtype", learning_rate=0.05, max_iter=2000, max_leaf_nodes=63,
            min_samples_leaf=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=0,
        )
        self.model_.fit(self.features(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.features(X))
