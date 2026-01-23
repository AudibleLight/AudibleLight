#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility variables, functions for downloading existing datasets"""

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


def combine_multizip(filename: str, destination: str, shell: bool = True) -> None:
    """
    Combine multiple zip files into one
    """
    subprocess.run(
        f"zip -s 0 {filename} --out {destination}",
        shell=shell,
    )


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


def download_txt(url: str) -> list[str]:
    """
    Download file from a given URL to a list of strings
    """
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful
    return response.text.splitlines()


def move_all_files_in_folder(
    src_folder: Union[str, Path], target_folder: Union[str, Path]
) -> None:
    """
    Move all files in a given folder to a new folder
    """

    src_files = os.listdir(src_folder)
    for file_name in src_files:
        full_file_name = os.path.join(src_folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, target_folder)


def increment_filename(pattern: str, upper_limit: int = 1000) -> str:
    """
    Increments the filename based on the pattern until an extension is found
    """
    i = 0

    while os.path.exists(pattern % i):
        i += 1
        if i > upper_limit:
            raise RuntimeError(f"Incremented filename too long: {pattern}")

    return pattern % i


class BaseDataSetup:
    """
    Base class for dataset setup
    """

    def __init__(self, dataset_home: str = None, metadata_path: str = None):
        self.dataset_home = dataset_home
        self.metadata_path = metadata_path

    def cleanup(self) -> None:
        print("Deleting source files that are not needed to use AudibleLight")

        # Remove files with these extensions
        multiparts = (f".z{str(i).zfill(2)}" for i in range(99))
        removers = (
            ".zip",
            ".csv",
            ".txt",
            ".navmesh",
            "checksums",
            ".pickle",
            *multiparts,
        )
        for root, dirs, files in os.walk(self.dataset_home):
            for file in files:
                if file.endswith(removers):
                    full_path = os.path.join(root, file)
                    os.remove(full_path)

        # Remove empty directories
        for root, dirs, files in os.walk(self.dataset_home, topdown=False):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                if not os.listdir(full_path):
                    shutil.rmtree(full_path)
