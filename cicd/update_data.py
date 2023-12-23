import os

from azure.storage.blob import BlobServiceClient

if __name__ == "__main__":
    connection_string = os.environ["STAGING_DATA_BLOB_CONNECTION_STRING"]
    staging_data_container_name = os.environ["STAGING_DATA_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(
        staging_data_container_name
    )
