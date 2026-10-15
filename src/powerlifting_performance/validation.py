import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate as sk_cross_validate

from powerlifting_performance.config import ID, TARGET


def duplicate_groups(train: pd.DataFrame) -> pd.Series:
    """One group id per set of rows with identical features.

    Identical rows are mostly the same performance entered in several divisions,
    so they must stay in the same fold.
    """
    features = [c for c in train.columns if c not in [ID, TARGET]]
    return train.groupby(features).ngroup()


def cross_validate(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   n_splits: int = 5, random_state: int = 0) -> pd.DataFrame:
    """RMSE and MAE (in kg) on each fold of a GroupKFold."""
    cv = GroupKFold(n_splits, shuffle=True, random_state=random_state)
    scores = sk_cross_validate(
        model, X, y, groups=groups, cv=cv,
        scoring={"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"},
    )
    return pd.DataFrame({"RMSE": -scores["test_RMSE"], "MAE": -scores["test_MAE"]})
