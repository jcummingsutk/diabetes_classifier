import os

from dotenv import load_dotenv


def load_env():
    env_path = "environment_variable_config.env"
    secrets_env_path = "environment_variable_config_secrets.env"
    load_dotenv(env_path)
    if os.path.exists(secrets_env_path):
        load_dotenv(secrets_env_path)
