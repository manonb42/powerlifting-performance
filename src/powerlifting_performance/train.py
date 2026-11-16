import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict

from powerlifting_performance.config import FEATURES, ID, TARGET
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.features import prepare_features
from powerlifting_performance.models import CatBoostExpert, GradientBoostingExpert, LevelLightGBM, TargetLightGBM
from powerlifting_performance.validation import cross_validate

EXPERTS = {
    "LightGBM": LevelLightGBM(),
    "LightGBM niveau relatif": TargetLightGBM("relative"),
    "LightGBM log": TargetLightGBM("log"),
    "Gradient boosting": GradientBoostingExpert(),
    "CatBoost": CatBoostExpert(level=False),
    "CatBoost niveau": CatBoostExpert(level=True),
}


def blend_with_twins(train: pd.DataFrame, test: pd.DataFrame, predictions: np.ndarray,
                     weight: float = 0.8) -> np.ndarray:
    """For the test rows identical to train rows on every feature (often the same performance entered in several
    divisions), predict weight * mean total of these train rows + (1 - weight) * the model's prediction.

    On the train rows with a twin, weight = 0.8 gives an RMSE of 36.5 kg, against 76.5 kg for the out-of-fold
    predictions alone and 37.0 kg for the twin total alone.
    This only helps when the test set was cut from the same data as train: new athletes have no twin.
    """
    twin_totals = train.groupby(FEATURES, as_index=False)[TARGET].mean()
    twin_total = test[FEATURES].merge(twin_totals, on=FEATURES, how="left")[TARGET].to_numpy()
    return np.where(np.isnan(twin_total), predictions, weight * twin_total + (1 - weight) * predictions)


def main():
    train, test = load_data()
    X, y, X_test = prepare_features(train, test)
    cv = GroupKFold(5, shuffle=True, random_state=0)

    oof, test_pred = {}, {}
    for name, model in EXPERTS.items():
        start = time.time()
        oof[name] = cross_val_predict(model, X, y, groups=X.duplicate_group, cv=cv)
        test_pred[name] = model.fit(X, y).predict(X_test)
        rmse = root_mean_squared_error(y, oof[name])
        print(f"{name:24s} RMSE hors-fold {rmse:.2f} ({(time.time() - start) / 60:.0f} min)", flush=True)

    P, P_test = np.column_stack(list(oof.values())), np.column_stack(list(test_pred.values()))
    stacking = LinearRegression(positive=True)
    scores = cross_validate(stacking, P, y, X.duplicate_group)
    print(f"Stacking : RMSE {scores.RMSE.mean():.2f} ± {scores.RMSE.std():.2f}")

    stacking.fit(P, y)
    for name, weight in zip(oof, stacking.coef_):
        print(f"  poids {name:24s} {weight:.3f}")
    predictions = blend_with_twins(train, test, stacking.predict(P_test))
    print(f"Soumission : {save_submission(test[ID], predictions, 'ensemble')}")


if __name__ == "__main__":
    main()
