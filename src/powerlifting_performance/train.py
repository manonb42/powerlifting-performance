"""Cross-validate a model, then fit it on the whole train set and write a submission.

Usage: uv run python -m powerlifting_performance.train [--model {lightgbm,gradient_boosting,linear}]
                                                       [--seeds N] [--twins] [--skip-cv]
"""

import argparse

from powerlifting_performance.config import ID
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.features import prepare, prepare_lightgbm, prepare_tree
from powerlifting_performance.model import LevelLightGBM, TotalKgRegressor, blend_with_twins, make_gradient_boosting
from powerlifting_performance.validation import cross_validate, duplicate_groups

MODELS = {
    "lightgbm": (prepare_lightgbm, LevelLightGBM, "lightgbm"),
    "gradient_boosting": (prepare_tree, make_gradient_boosting, "gradient_boosting"),
    "linear": (prepare, TotalKgRegressor, "regression_lineaire"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="lightgbm")
    parser.add_argument("--seeds", type=int, default=1, help="lightgbm only: number of models averaged")
    parser.add_argument("--twins", action="store_true",
                        help="blend the submission with the totals of exact duplicates in train")
    parser.add_argument("--skip-cv", action="store_true", help="only fit and write the submission")
    args = parser.parse_args()
    prepare_fn, make_model, submission_name = MODELS[args.model]

    train, test = load_data()
    X, y, X_test = prepare_fn(train, test)
    model = make_model()
    if args.model == "lightgbm":
        model.set_params(n_seeds=args.seeds)
        submission_name += f"_{args.seeds}graines"

    if not args.skip_cv:
        scores = cross_validate(model, X, y, duplicate_groups(train))
        print(scores.round(2))
        print(f"RMSE : {scores.RMSE.mean():.2f} ± {scores.RMSE.std():.2f}")
        print(f"MAE  : {scores.MAE.mean():.2f}")

    predictions = model.fit(X, y).predict(X_test)
    if args.twins:
        predictions = blend_with_twins(train, test, predictions)
        submission_name += "_doublons"
    path = save_submission(test[ID], predictions, submission_name)
    print(f"Soumission : {path}")


if __name__ == "__main__":
    main()
