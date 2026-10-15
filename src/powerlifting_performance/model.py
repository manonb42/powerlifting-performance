import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from powerlifting_performance.config import CAT_COLS


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
