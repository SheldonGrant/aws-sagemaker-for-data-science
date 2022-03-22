import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    return data

if __name__ == "__main__":

    data = get_data()

    df_train, df_test = train_test_split(
        data, test_size=0.33, random_state=42
    )

    df_train.to_csv("data/train/data.csv")
    df_test.to_csv("data/test/data.csv")
