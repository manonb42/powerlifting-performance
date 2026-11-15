import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from powerlifting_performance.model import LevelLightGBM


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
