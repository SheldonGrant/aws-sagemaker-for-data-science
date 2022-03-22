import os
import argparse
from pathlib import Path
import pandas as pd
from sklearn.svm import SVC
import joblib


def get_data(train_path, test_path):

    df_train = pd.read_csv(train_path, engine="python")

    df_test = pd.read_csv(test_path, engine="python")

    return df_train, df_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training script")

    # important paths
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="path where training data is found.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ["SM_CHANNEL_TEST"],
        help="path where testing data is found.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="path to model artifacts directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DIR"],
        help="path to output artifacts directory",
    )

    # hyperparameters
    # passed to script at run time
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--n_jobs", type=int)

    args = parser.parse_args()

    model_path = Path(args.model_dir, "model.joblib").as_posix()
    # query_data from athena
    df_train, df_test = get_data(
        train_path=Path(args.train, "data.csv").as_posix(),
        test_path=Path(args.test, "data.csv").as_posix()
    )

    # preprocessing for the algorithm
    # X =
    # y =

    # data, target split
    X_train, y_train = df_train.drop(["target"], axis=1), df_train["target"]
    X_test, y_test = df_test.drop(["target"], axis=1), df_test["target"]

    # train model
    clf = SVC()
    clf.fit(X_train, y_train)

    print(f"score (accuracy) = {clf.score(X_test, y_test)}")

    joblib.dump(clf, model_path)
