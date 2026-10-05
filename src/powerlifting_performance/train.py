"""Cross-validate the linear model, then fit it on the whole train set and write a submission.

Usage: uv run python -m powerlifting_performance.train
"""

from powerlifting_performance.config import ID
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.features import prepare
from powerlifting_performance.model import TotalKgRegressor
from powerlifting_performance.validation import cross_validate, duplicate_groups


def main():
    train, test = load_data()
    X, y, X_test = prepare(train, test)
    model = TotalKgRegressor()

    scores = cross_validate(model, X, y, duplicate_groups(train))
    print(scores.round(2))
    print(f"RMSE : {scores.RMSE.mean():.2f} ± {scores.RMSE.std():.2f}")
    print(f"MAE  : {scores.MAE.mean():.2f}")

    model.fit(X, y)
    path = save_submission(test[ID], model.predict(X_test), "regression_lineaire")
    print(f"Soumission : {path}")


if __name__ == "__main__":
    main()
