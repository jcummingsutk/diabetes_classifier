import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    print(params)
    training_percentage = params["xgboost_training"]["training_percentage"]

    processed_data_file = os.path.join("data", "processed", "diabetes.csv")
    data = pd.read_csv(processed_data_file)

    df_train, df_test = train_test_split(
        data, random_state=42, train_size=training_percentage
    )

    out_training_file = os.path.join("data", "processed", "train.pkl")
    out_testing_file = os.path.join("data", "processed", "test.pkl")

    df_train.to_pickle(out_training_file)
    df_test.to_pickle(out_testing_file)
