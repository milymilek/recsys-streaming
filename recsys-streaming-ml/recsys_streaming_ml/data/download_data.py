import requests
import pathlib
import tarfile
import gzip
import shutil

from recsys_streaming_ml.config import (DOWNLOAD_DATA_URL,
                                        DOWNLOAD_METADATA_URL,
                                        DATA_FILE,
                                        METADATA_FILE)


def _download_data(url: str, filepath: pathlib.Path) -> None:
    response = requests.get(url)

    with open(filepath.with_suffix(".gz"), 'wb') as f:
        f.write(response.content)


def _untar_data(filepath: pathlib.Path) -> None:
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=filepath.parent) 


def _unzip_data(filepath: pathlib.Path) -> None:
    with gzip.open(filepath.with_suffix(".gz"), 'rb') as f_in:
        with open(filepath.with_suffix(".jsonl"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def _delete_data(filepath: pathlib.Path) -> None:
    filepath.with_suffix(".gz").unlink()


def run():
    print("SCRIPT: Download data - START")

    for url, datafile in zip([DOWNLOAD_DATA_URL, DOWNLOAD_METADATA_URL], [DATA_FILE, METADATA_FILE]):
        _download_data(url=url, filepath=datafile)
        _unzip_data(filepath=datafile)
        _delete_data(filepath=datafile)

    print(f'Files downloaded, extracted and deleted.')
    print("SCRIPT: Download data - END")
