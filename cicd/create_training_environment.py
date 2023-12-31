import os

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import DockerBuildContext, Environment

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
    env_name = os.environ["TRAINING_ENVIRONMENT_NAME"]
    build = DockerBuildContext.from_local_directory(
        workspace=ws, path="./docker_model_build_context/"
    )
    env = Environment.from_docker_build_context(
        name=env_name, docker_build_context=build
    )
    build_details = env.build(workspace=ws)
    build_details.wait_for_completion()
