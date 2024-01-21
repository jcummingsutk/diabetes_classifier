import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.identity import EnvironmentCredential
from azure.storage.blob import BlobClient, BlobServiceClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from diabetes_classifier.env_vars import load_env

if __name__ == "__main__":
    load_env()
    # Rename some environment variables for the EnvironmentCredential
    os.environ["AZURE_TENANT_ID"] = os.environ["SERVICE_PRINCIPAL_TENANT_ID"]
    os.environ["AZURE_CLIENT_ID"] = os.environ["SERVICE_PRINCPAL_CLIENT_ID"]
    os.environ["AZURE_CLIENT_SECRET"] = os.environ["SERVICE_PRINCIPAL_CLIENT_SECRET"]
    staging_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["STAGING_AZURE_ML_WORKSPACE_NAME"],
        credential=EnvironmentCredential(),
    )

    # Get all the jobs that are from batch deployments, sort them by datetime,
    # then get the latest one

    job_list: list[tuple[Job, datetime]] = [
        (j, j.creation_context.created_at)
        for j in staging_client.jobs.list()
        if j.status == "Completed" and j.tags["azureml.batchrun"] == "true"
    ]
    job_list = sorted(job_list, key=lambda j_tuple: j_tuple[-1])
    latest_job = job_list[-1][0]

    # Get the predictions
    staging_client.jobs.download(
        latest_job.name,
        download_path=".",
        output_name="score",
    )

    # Create and save the figures
    df_predictions = pd.read_csv("data_pred.csv")
    y_true = df_predictions["Diabetes"].to_list()
    y_pred = df_predictions["prediction"].to_list()

    os.makedirs("images", exist_ok=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"]
    )
    disp.plot(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "confusion_matrix.png"))

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "No Diabetes",
            "Diabetes",
        ],
        output_dict=True,
    )
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    fig.tight_layout()
    fig = sns.heatmap(
        pd.DataFrame(
            {
                key: report[key]
                for key in ["No Diabetes", "Diabetes", "accuracy", "weighted avg"]
            }
        ).T,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1.0,
    )
    plt.savefig(os.path.join("images", "classification_report.png"))

    # Upload images to public storage for my work sample
    connection_string = os.environ["DEV_DATA_BLOB_CONNECTION_STRING"]
    dev_data_container_name = os.environ["PUBLIC_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(dev_data_container_name)
    blobs = [
        os.path.join("images", "classification_report.png"),
        os.path.join("images", "confusion_matrix.png"),
    ]
    for blob_filename in blobs:
        print(blob_filename)
        with open(blob_filename, "rb") as f:
            blob_client = BlobClient.from_connection_string(
                conn_str=connection_string,
                container_name=dev_data_container_name,
                blob_name=blob_filename,
                max_block_size=1024 * 1024 * 4,
                max_single_put_size=1024 * 1024 * 8,
            )
            blob_client.upload_blob(f, overwrite=True, blob_type="BlockBlob")
