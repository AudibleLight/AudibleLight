#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and prepares the waypoints for the Gibson Environment dataset"""

import argparse
import os
import shutil
from pathlib import Path

from audiblelight import utils
from scripts.download_data.utils import download_file, extract_tar

# Download paths
WAYPOINTS_ZIP = (
    "https://storage.googleapis.com/gibson_scenes/navigation_scenarios.tar.gz"
)

DEFAULT_PATH = str(utils.get_project_root() / "resources/waypoints")
if not os.path.exists(DEFAULT_PATH):
    os.makedirs(DEFAULT_PATH)

DEFAULT_CLEANUP = True


def main(
    path: str = DEFAULT_PATH,
    cleanup: bool = DEFAULT_CLEANUP,
) -> None:
    f"""
    Downloads and prepares the waypoints for the Gibson Environment dataset. NOTE: by running this script, we assume that you have completed the user agreement form (https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform).

    Arguments:
        path: Path to store and process the dataset, defaults to {DEFAULT_PATH}.
        cleanup: Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}.
    """
    print("---- Gibson Environment Database waypoints download script ----")
    print(f"Waypoints will be downloaded to: {path}")
    path = Path(path)

    print(f"Downloading {WAYPOINTS_ZIP}...")
    tar_path = path / "waypoints.tar.gz"
    download_file(WAYPOINTS_ZIP, tar_path)
    print(f"... downloaded to {tar_path}")

    extract_tar(tar_path, path)
    print(f"... extracted to {path}")

    # only keep full+ folder, that contains all the data
    full_plus_path = path / "navigation_scenarios/waypoints/full+"

    gib_path = path / "gibson"
    if not os.path.exists(gib_path):
        os.makedirs(gib_path)

    for f in full_plus_path.glob("*.json"):
        f.rename(gib_path / f.name)

    if cleanup:
        os.remove(tar_path)
        print(f"... removed {tar_path}")

        shutil.rmtree(path / "navigation_scenarios")


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
