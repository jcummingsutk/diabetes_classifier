import os

from azure.storage.blob import BlobClient, BlobServiceClient

if __name__ == "__main__":
    connection_string = os.environ["STAGING_DATA_BLOB_CONNECTION_STRING"]
    staging_data_container_name = os.environ["STAGING_DATA_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(
        staging_data_container_name
    )
    blobs = [
        os.path.join("data", "raw", "diabetes.csv"),
        os.path.join("data", "processed", "diabetes.csv"),
        os.path.join("data", "processed", "train.pkl"),
        os.path.join("data", "processed", "test.pkl"),
    ]
    for blob_filename in blobs:
        print(blob_filename)
        with open(blob_filename, "rb") as f:
            # blob_client = container_client.get_blob_client(blob_filename)
            blob_client = BlobClient.from_connection_string(
                conn_str=connection_string,
                container_name=staging_data_container_name,
                blob_name=blob_filename,
                max_block_size=1024 * 1024 * 4,
                max_single_put_size=1024 * 1024 * 8,
            )
            blob_client.upload_blob(f, overwrite=True, blob_type="BlockBlob")
