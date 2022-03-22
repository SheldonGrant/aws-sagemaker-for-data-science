import argparse
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

def get_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    df_train, df_test = train_test_split(
        data, test_size=0.33, random_state=42
    )

    return df_train, df_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run training job')
    parser.add_argument('--model', default='model.joblib', type=str)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--n_jobs', type=int)
    args = parser.parse_args()

    model_path = f"/opt/ml/model/{args.model}"

    # query_data from athena
    df_train, df_test = get_data()

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

    joblib.dump(clf, args.model)
