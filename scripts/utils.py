#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility variables, functions for downloading existing datasets"""

import tarfile
import zipfile

import requests
from tqdm import tqdm


def extract_tar(tar_path: str, destination: str) -> None:
    """
    Extracts a tar file to the given destination.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(destination)


def extract_zip(zip_path: str, destination: str) -> None:
    """
    Extracts a zip file to the given destination.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)
    except zipfile.BadZipFile:
        raise ValueError("The provided file is not a valid zip file.")


def download_file(url: str, destination: str, block_size: int = 1024) -> None:
    """
    Downloads a file from a URL to a local destination.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
        with open(destination, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download file: {e}")
