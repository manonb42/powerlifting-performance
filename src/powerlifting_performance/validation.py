import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate as sk_cross_validate


def cross_validate(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = 0) -> pd.DataFrame:
    """RMSE and MAE (in kg) on each fold of a 5-fold GroupKFold."""
    cv = GroupKFold(5, shuffle=True, random_state=random_state)
    scores = sk_cross_validate(
        model, X, y, groups=groups, cv=cv,
        scoring={"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"},
    )
    return pd.DataFrame({"RMSE": -scores["test_RMSE"], "MAE": -scores["test_MAE"]})
