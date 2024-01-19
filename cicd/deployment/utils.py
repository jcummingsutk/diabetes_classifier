import pandas as pd


def get_latest_run(df: pd.DataFrame) -> str:
    df = df.sort_values("start_time")
    ids = df["run_id"].to_list()
    if len(ids) == 0:
        raise ValueError("Inputted DataFrame is Empty")
    return ids[-1]


def get_best_run_id(df: pd.DataFrame, metric_column: str) -> tuple[str, str]:
    if df.shape[0] == 0:
        raise ValueError("Inputted DataFrame is Empty")
    best_run = df.sort_values(metric_column, ascending=False).iloc[0]
    return best_run.run_id
