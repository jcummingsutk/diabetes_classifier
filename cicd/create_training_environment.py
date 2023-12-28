import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential

if __name__ == "__main__":
    subscription_id = os.environ["AZURE_ML_SUBSCRIPTION_ID"]
    resource_group_name = os.environ["AZURE_ML_RESOURCE_GROUP_NAME"]
    workspace_name = os.environ["AZURE_ML_WORKSPACE_NAME"]

    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        credential=DefaultAzureCredential(),
    )
    build = BuildContext(path="./docker_build_context/")
    env_docker_context = Environment(build=build, name="diabetes-classifier-staging")
    ml_client.environments.create_or_update(env_docker_context)
