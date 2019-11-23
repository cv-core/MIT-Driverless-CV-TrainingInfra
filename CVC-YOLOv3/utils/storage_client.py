import os
import requests
import pathlib
import shutil

from google.cloud import storage
from retrying import retry

gc_storage_client = storage.Client(project="mitdriverless")
local_dir = os.path.join(os.path.dirname(__file__), "gs")
pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)

def get_uri_filepath(uri):
    return os.path.join(local_dir, uri.replace("/", "_").lower())

@retry(stop_max_attempt_number=3)
def delete_file(uri):
    if uri.startswith("gs://"):
        uri_parts = uri.split("/")
        bucket_name = uri_parts[2]
        bucket = gc_storage_client.get_bucket(bucket_name)
        blob = bucket.blob("/".join(uri_parts[3:]))
        blob.delete()
        os.remove(get_uri_filepath(uri))
        return
    raise BaseException("unsupported uri scheme. must begin with gs://", uri)

@retry(stop_max_attempt_number=3)
def get_file(uri, use_cache=True):
    uri_filepath = get_uri_filepath(uri)
    if os.path.exists(uri_filepath) and use_cache:
        return uri_filepath
    if uri.startswith("gs://"):
        uri_parts = uri.split("/")
        bucket_name = uri_parts[2]
        bucket = gc_storage_client.get_bucket(bucket_name)
        blob = bucket.blob("/".join(uri_parts[3:]))
        # using this '.tmp' trick so partially downloaded files won't corrupt.
        tmp_path = uri_filepath + ".tmp"
        blob.download_to_filename(tmp_path)
        os.rename(tmp_path, uri_filepath)
        return uri_filepath
    if uri.startswith("http://") or uri.startswith("https://"):
        response = requests.get(uri, stream=True)
        response.raise_for_status()
        with open(uri_filepath, 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)
        return uri_filepath
    if os.path.isfile(uri):
        return uri
    raise BaseException("unsupported uri scheme. must begin with http://, https://, or gs://, or be a local file", uri)

@retry(stop_max_attempt_number=3)
def upload_file(filepath, uri, use_cache=True):
    # print("filepath: ", filepath)
    # print("uri: ", uri)
    if uri.startswith("gs://"):
        uri_parts = uri.split("/")
        bucket_name = uri_parts[2]
        bucket = gc_storage_client.get_bucket(bucket_name)
        # create the intermediate folders leading up to the file.
        # That way, gcsfuse will work correctly. See
        # https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/semantics.md#files-and-directories
        for i in range(3, len(uri_parts)):
            intermediate_folder = "/".join(uri_parts[3:i]) + "/"
            blob = bucket.blob(intermediate_folder)
            if not blob.exists():
                blob.upload_from_string(b'')
        blob = bucket.blob("/".join(uri_parts[3:]))
        blob.upload_from_filename(filepath)
        if use_cache:
            shutil.copy(filepath, get_uri_filepath(uri))
        return
    raise BaseException("unsupported uri scheme. must begin with gs://", uri)
