import os

import mlflow
from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import (
    BatchEndpoint,
    BatchRetrySettings,
    CodeConfiguration,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
)
from azure.ai.ml.exceptions import ValidationException
from azure.identity import EnvironmentCredential
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

from cicd.deployment.utils import get_best_run_id, get_latest_run
from diabetes_classifier.env_vars import load_env

if __name__ == "__main__":
    load_env()

    # Rename some environment variables for the EnvironmentCredential
    os.environ["AZURE_TENANT_ID"] = os.environ["SERVICE_PRINCIPAL_TENANT_ID"]
    os.environ["AZURE_CLIENT_ID"] = os.environ["SERVICE_PRINCPAL_CLIENT_ID"]
    os.environ["AZURE_CLIENT_SECRET"] = os.environ["SERVICE_PRINCIPAL_CLIENT_SECRET"]

    # Construct clients
    dev_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["DEV_AZURE_ML_WORKSPACE_NAME"],
        credential=EnvironmentCredential(),
    )
    staging_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["STAGING_AZURE_ML_WORKSPACE_NAME"],
        credential=EnvironmentCredential(),
    )
    registry_client = MLClient(
        credential=EnvironmentCredential(),
        registry_name="diabetes-classifier",
        registry_location="us-east2",
    )

    # Set mlflow URI
    mlflow_tracking_uri = dev_client.workspaces.get(
        dev_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Training runs have parents where the children are the best N when tuning
    # hyperparameters. Get the best, most recent model by
    # 1) getting the most recent run that is a parent, then
    # 2) getting the child that has the best metric
    # We'll then register the model in the dev workspace

    all_parent_runs = mlflow.search_runs(
        experiment_names=["diabetes_prediction"],
        filter_string="tags.isParentRun = '1'",
    )

    latest_run = get_latest_run(all_parent_runs)

    runs_with_parent = mlflow.search_runs(
        experiment_names=["diabetes_prediction"],
        filter_string=f"tags.mlflow.parentRunId = '{latest_run}'",
    )

    best_run_id = get_best_run_id(runs_with_parent, "metrics.f_score_eval")
    registered_model = mlflow.register_model(
        f"runs:/{best_run_id}/model/", name="diabetes-classifier"
    )

    # Now that it is in the dev workspace, we will look into the registry for the
    # latest version of the model, and we'll up the version, then share the best
    # model in the dev space with the registry.
    try:
        latest_version_str = registry_client.models._get_latest_version(
            registered_model.name
        ).version
    except ValidationException:
        latest_version_str = "0"

    next_version = str(int(latest_version_str) + 1)

    model_in_registry = dev_client.models.share(
        name=registered_model.name,
        version=registered_model.version,
        share_with_name=registered_model.name,
        share_with_version=next_version,
        registry_name="diabetes-classifier",
    )

    # Setup staging sdk v1 creds
    sp_auth = ServicePrincipalAuthentication(
        tenant_id=os.environ["SERVICE_PRINCIPAL_TENANT_ID"],
        service_principal_id=os.environ["SERVICE_PRINCPAL_CLIENT_ID"],
        service_principal_password=os.environ["SERVICE_PRINCIPAL_CLIENT_SECRET"],
    )
    ws = Workspace(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["STAGING_AZURE_ML_WORKSPACE_NAME"],
        auth=sp_auth,
    )
    env = staging_client.environments._get_latest_version("diabetes-classifier-test")

    # Create the endpoint
    endpoint_name = "diabetes-classifier-batch"
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description="",
        tags={},
    )
    endpoint_result = staging_client.begin_create_or_update(endpoint).result()

    # Create the deployment
    deployment = ModelBatchDeployment(
        name=endpoint_result.name,
        description="",
        endpoint_name=endpoint_name,
        model=model_in_registry,
        code_configuration=CodeConfiguration(
            code=os.path.join("cicd", "deployment", "scoring"),
            scoring_script="batch_driver.py",
        ),
        environment=env,
        compute="diabetes-classifier-staging",
        settings=ModelBatchDeploymentSettings(
            max_concurrency_per_instance=2,
            mini_batch_size=10,
            instance_count=2,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name="predictions.csv",
            retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
            logging_level="info",
        ),
    )
    deployment_result = staging_client.begin_create_or_update(deployment).result()

    endpoint = staging_client.batch_endpoints.get(endpoint_name)
    endpoint.defaults.deployment_name = deployment_result.name
    staging_client.batch_endpoints.begin_create_or_update(endpoint).result()

    # Invoke the endpoint
    job = staging_client.batch_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment.name,
        input=Input(
            path="azureml://datastores/staging_diabetes_data/paths/data/staging/batch/data.csv",
            type=AssetTypes.URI_FILE,
        ),
    )
    staging_client.jobs.stream(job.name)

    # Clean up the endpoint
    staging_client.batch_endpoints.begin_delete(name=endpoint.name)
