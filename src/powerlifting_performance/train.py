import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict

from powerlifting_performance.config import ID
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.experts import CatBoostExpert, GradientBoostingExpert, TargetLightGBM
from powerlifting_performance.features import prepare_features
from powerlifting_performance.model import LevelLightGBM, blend_with_twins
from powerlifting_performance.validation import cross_validate

EXPERTS = {
    "LightGBM": LevelLightGBM(),
    "LightGBM niveau relatif": TargetLightGBM("relative"),
    "LightGBM log": TargetLightGBM("log"),
    "Gradient boosting": GradientBoostingExpert(),
    "CatBoost": CatBoostExpert(level=False),
    "CatBoost niveau": CatBoostExpert(level=True),
}


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
