"""Cross-validate a model, then fit it on the whole train set and write a submission.

Usage: uv run python -m powerlifting_performance.train [--model {gradient_boosting,linear}]
"""

import argparse

from powerlifting_performance.config import ID
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.features import prepare, prepare_tree
from powerlifting_performance.model import TotalKgRegressor, make_gradient_boosting
from powerlifting_performance.validation import cross_validate, duplicate_groups

MODELS = {
    "gradient_boosting": (prepare_tree, make_gradient_boosting, "gradient_boosting"),
    "linear": (prepare, TotalKgRegressor, "regression_lineaire"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="gradient_boosting")
    args = parser.parse_args()
    prepare_fn, make_model, submission_name = MODELS[args.model]

    train, test = load_data()
    X, y, X_test = prepare_fn(train, test)
    model = make_model()

    scores = cross_validate(model, X, y, duplicate_groups(train))
    print(scores.round(2))
    print(f"RMSE : {scores.RMSE.mean():.2f} ± {scores.RMSE.std():.2f}")
    print(f"MAE  : {scores.MAE.mean():.2f}")

    predictions = model.fit(X, y).predict(X_test)
    path = save_submission(test[ID], predictions, submission_name)
    print(f"Soumission : {path}")


if __name__ == "__main__":
    main()
