import argparse

from powerlifting_performance.config import ID
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.features import prepare_lightgbm
from powerlifting_performance.model import LevelLightGBM, blend_with_twins
from powerlifting_performance.validation import cross_validate, duplicate_groups


def main():
    parser = argparse.ArgumentParser(description="Cross-validate the model, fit it on train and write a submission.")
    parser.add_argument("--seeds", type=int, default=1, help="number of models averaged")
    parser.add_argument("--twins", action="store_true",
                        help="blend the submission with the totals of exact duplicates in train")
    parser.add_argument("--skip-cv", action="store_true", help="only fit and write the submission")
    args = parser.parse_args()

    train, test = load_data()
    X, y, X_test = prepare_lightgbm(train, test)
    model = LevelLightGBM(n_seeds=args.seeds)

    if not args.skip_cv:
        scores = cross_validate(model, X, y, duplicate_groups(train))
        print(scores.round(2))
        print(f"RMSE : {scores.RMSE.mean():.2f} ± {scores.RMSE.std():.2f}")
        print(f"MAE  : {scores.MAE.mean():.2f}")

    predictions = model.fit(X, y).predict(X_test)
    name = f"lightgbm_{args.seeds}graines"
    if args.twins:
        predictions = blend_with_twins(train, test, predictions)
        name += "_doublons"
    print(f"Soumission : {save_submission(test[ID], predictions, name)}")


if __name__ == "__main__":
    main()
