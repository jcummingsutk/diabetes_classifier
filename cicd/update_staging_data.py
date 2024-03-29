import os

from azure.storage.blob import BlobClient, BlobServiceClient

from diabetes_classifier.env_vars import load_env

if __name__ == "__main__":
    load_env()
    connection_string = os.environ["DEV_DATA_BLOB_CONNECTION_STRING"]
    dev_data_container_name = os.environ["STAGING_DATA_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(dev_data_container_name)
    blobs = [
        os.path.join("data", "staging", "batch", "data.csv"),
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
