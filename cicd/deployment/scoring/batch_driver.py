import glob
import os
from pathlib import Path

import pandas as pd
from mlflow.pyfunc import load_model


def init():
    global model
    global output_path
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # The path "model" is the name of the registered model's folder
    output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]
    model_path = glob.glob(os.environ["AZUREML_MODEL_DIR"] + "/*/")[0]

    # load the model
    model = load_model(model_path)


def run(mini_batch):
    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        print("data")
        print(data.info())
        print("data diabetes dropped:")
        print(data.drop(["Diabetes"], axis=1))
        pred = model.predict(data.drop(["Diabetes"], axis=1))
        data["prediction"] = pred

        output_file_name = Path(file_path).stem
        output_file_path = os.path.join(output_path, output_file_name + "_pred.csv")
        data.to_csv(output_file_path)
    return mini_batch
