from powerlifting_performance.config import ID
from powerlifting_performance.data import load_data, save_submission
from powerlifting_performance.features import prepare_lightgbm
from powerlifting_performance.model import LevelLightGBM, blend_with_twins
from powerlifting_performance.validation import cross_validate, duplicate_groups


def main():
    train, test = load_data()
    X, y, X_test = prepare_lightgbm(train, test)
    model = LevelLightGBM()

    scores = cross_validate(model, X, y, duplicate_groups(train))
    print(scores.round(2))
    print(f"RMSE : {scores.RMSE.mean():.2f} ± {scores.RMSE.std():.2f}")
    print(f"MAE  : {scores.MAE.mean():.2f}")

    predictions = blend_with_twins(train, test, model.fit(X, y).predict(X_test))
    print(f"Soumission : {save_submission(test[ID], predictions, 'lightgbm')}")


if __name__ == "__main__":
    main()
