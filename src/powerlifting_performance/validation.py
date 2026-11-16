import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate as sk_cross_validate


def cross_validate(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = 0) -> pd.DataFrame:
    """Cross-validate a model on 5 folds, exact duplicates kept together.

    Args:
        model: Model to evaluate.
        X: Features.
        y: Totals, in kg.
        groups: Duplicate group of each row.
        random_state: Seed of the fold split.

    Returns:
        RMSE and MAE in kg, one row per fold.
    """
    cv = GroupKFold(5, shuffle=True, random_state=random_state)
    scores = sk_cross_validate(
        model, X, y, groups=groups, cv=cv,
        scoring={"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"},
    )
    return pd.DataFrame({"RMSE": -scores["test_RMSE"], "MAE": -scores["test_MAE"]})
