import pandas as pd

from cicd.deployment.utils import get_latest_run


def test_latest_runs():
    test_input = pd.DataFrame(
        {
            "run_id": ["127", "122", "111"],
            "start_time": [
                pd.to_datetime("2022-01-01", utc=True),
                pd.to_datetime("2022-01-02", utc=True),
                pd.to_datetime("2022-01-14", utc=True),
            ],
        }
    )

    latest_run = get_latest_run(test_input)
    assert latest_run == "111"
