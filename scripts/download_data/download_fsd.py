#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and prepares the FSD50k."""

import argparse
import os
import shutil
import subprocess

from audiblelight.utils import get_project_root
from scripts.download_data.utils import BaseDataSetup, download_txt

try:
    import soundata
    import soundata.download_utils
except ImportError:
    raise ImportError(
        "'soundata' is required to download FSD50K; install it with 'pip install soundata'"
    )


DEFAULT_PATH = str(get_project_root() / "resources/soundevents")
DEFAULT_CLEANUP = False
DCASE_FSD50K_SELECTED = "https://zenodo.org/record/6406873/files/FSD50K_selected.txt"


def download_multipart_zip(zip_remotes, save_dir, force_overwrite, cleanup):
    """
    Download and unzip a multipart zip file.
    """
    for l_ in range(len(zip_remotes)):
        soundata.download_utils.download_from_remote(
            zip_remotes[l_], save_dir, force_overwrite
        )
    zip_path = os.path.join(
        save_dir,
        next((part.filename for part in zip_remotes if ".zip" in part.filename), None),
    )
    out_path = zip_path.replace(".zip", "_single.zip")
    subprocess.run(["zip", "-s", "0", zip_path, "--out", out_path])
    if cleanup:
        for l_ in range(len(zip_remotes)):
            zip_path = os.path.join(save_dir, zip_remotes[l_].filename)
            os.remove(zip_path)
    soundata.download_utils.unzip(out_path, cleanup=cleanup)


# Patch soundata multipart zip function
soundata.download_utils.download_multipart_zip = download_multipart_zip


class FSD50KDataSetup(BaseDataSetup):
    def __init__(self, dataset_name: str = "fsd50k", download: bool = True, **kwargs):
        """Initializes the FSD50K dataset setup."""
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.download = download
        self.url_fsd_selected_txt = DCASE_FSD50K_SELECTED

        # Filled in later
        self.fsd50k = None

    def download_dataset(self):
        """Downloads the FSD50K dataset."""
        if self.download:
            try:
                self.fsd50k.download()
            except Exception as e:
                raise RuntimeError(f"Failed to download FSD50K dataset: {e}")

    def prepare_dataset(self):
        """Prepares the FSD50K dataset by initializing and possibly downloading it."""
        if self.dataset_home is None and self.download:
            raise ValueError(
                "Dataset home path must be provided when download is enabled."
            )
        self.fsd50k = soundata.initialize(self.dataset_name, self.dataset_home)
        self.download_dataset()

    def to_dcase_format(self, target_subdir: str = DEFAULT_PATH):
        # Download the lines from the URL
        lines = download_txt(self.url_fsd_selected_txt)

        # Loop through each line in the text
        for line in lines:
            new_dir = os.path.dirname(line.strip())
            filename = os.path.basename(line.strip())

            # Retrieve the new source directory for the file
            if "train" in new_dir:
                source = os.path.join(self.dataset_home, "FSD50K.dev_audio")
            elif "test" in new_dir:
                source = os.path.join(self.dataset_home, "FSD50K.eval_audio")
            else:
                raise ValueError("Invalid directory structure in the text file")

            # Full path of the source file
            src_file = os.path.join(source, filename)
            # New directory, check if it doesn't exist
            dest_path = os.path.join(self.dataset_home, target_subdir)
            os.makedirs(os.path.join(dest_path, new_dir), exist_ok=True)
            # Copy file to new directory
            shutil.copy(src_file, os.path.join(dest_path, new_dir, filename))


def main(
    path: str = DEFAULT_PATH,
    cleanup: bool = DEFAULT_CLEANUP,
):
    f"""
    Downloads and prepares the FSD50K dataset.

    Arguments:
        path: Path to store and process the dataset, defaults to {DEFAULT_PATH}.
        cleanup: Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}.
    """
    print("---- FSD50K download script ----")
    print(f"Datasets will be downloaded to: {path}")

    fsd50k = FSD50KDataSetup(
        dataset_name="fsd50k",
        download=True,
        dataset_home=path,
    )
    fsd50k.prepare_dataset()
    fsd50k.to_dcase_format()
    if cleanup:
        fsd50k.cleanup()

        # Remove additional folders we've created
        #  Try block is needed as we might have removed these inside `cleanup` method
        dirs = [
            "FSD50K.dev_audio",
            "FSD50K.eval_audio",
            "FSD50K.doc",
            "FSD50K.metadata",
            "FSD50K.ground_truth",
        ]
        for d in dirs:
            try:
                shutil.rmtree(os.path.join(path, d))
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to store and process the dataset, defaults to {DEFAULT_PATH}",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=f"Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}",
        default=DEFAULT_CLEANUP,
    )
    args = vars(parser.parse_args())
    main(**args)
