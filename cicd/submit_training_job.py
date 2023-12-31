import os

from azure.ai.ml import Input, MLClient, command
from azure.identity import EnvironmentCredential
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment

from diabetes_classifier.env_vars import load_env

if __name__ == "__main__":
    load_env()
    sp_auth = ServicePrincipalAuthentication(
        tenant_id=os.environ["SERVICE_PRINCIPAL_TENANT_ID"],
        service_principal_id=os.environ["SERVICE_PRINCPAL_CLIENT_ID"],
        service_principal_password=os.environ["SERVICE_PRINCIPAL_CLIENT_SECRET"],
    )
    ws = Workspace(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
        auth=sp_auth,
    )

    # Rename some environment variables for the EnvironmentCredential
    os.environ["AZURE_TENANT_ID"] = os.environ["SERVICE_PRINCIPAL_TENANT_ID"]
    os.environ["AZURE_CLIENT_ID"] = os.environ["SERVICE_PRINCPAL_CLIENT_ID"]
    os.environ["AZURE_CLIENT_SECRET"] = os.environ["SERVICE_PRINCIPAL_CLIENT_SECRET"]
    ml_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
        credential=EnvironmentCredential(),
    )
    env_version = Environment.list(ws)[os.environ["TRAINING_ENVIRONMENT_NAME"]].version
    print(f"using environment version {env_version}")

    command_job = command(
        code=os.path.join(
            "diabetes_classifier",
            "diabetes_classifier_models",
            "diabetes_classifier_models",
            "xgboost",
        ),
        command="python train.py --num-folds 5 --train-data ${{inputs.train_data}} --test-data ${{inputs.test_data}} --num-hyperopt-evals 1000 --num-hyperopt-trials-to-log 20 --target-column Diabetes",
        environment=f"{os.environ['TRAINING_ENVIRONMENT_NAME']}:{env_version}",
        inputs={
            "train_data": Input(
                type="uri_file",
                path="azureml://subscriptions/94f3bfe4-d65b-4af2-959a-f4cc3f4fef6a/resourcegroups/john-cummings/workspaces/john-cummings-ml/datastores/staging_diabetes_data/paths/data/processed/train.pkl",
            ),
            "test_data": Input(
                type="uri_file",
                path="azureml://subscriptions/94f3bfe4-d65b-4af2-959a-f4cc3f4fef6a/resourcegroups/john-cummings/workspaces/john-cummings-ml/datastores/staging_diabetes_data/paths/data/processed/test.pkl",
            ),
        },
        compute="diabetes-staging-cluster",
        experiment_name="diabetes_prediction",
    )
    returned_job = ml_client.jobs.create_or_update(command_job)
    ml_client.jobs.stream(returned_job.name)
