import os

import pandas as pd


def process_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    for col in df_raw.columns:
        df_raw[col] = df_raw[col].astype(int)
    df_raw.rename(columns={"Diabetes_binary": "Diabetes"}, inplace=True)
    output_data_file = os.path.join("data", "processed", "diabetes.csv")
    df_raw.to_csv(output_data_file)
    return df_raw


if __name__ == "__main__":
    raw_data_file = os.path.join("data", "raw", "diabetes.csv")
    df_raw = pd.read_csv(raw_data_file)
    process_data(df_raw)
