#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and prepares the Gibson Environment dataset meshes.

NOTE: by running this script, we assume that you have completed the user agreement form (https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform).
"""

import argparse
import os
import sys
from pathlib import Path

from audiblelight import utils
from scripts.download_data.utils import download_file, extract_zip

USER_AGREEMENT = "https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform"
# Download paths: the former contains 4 scenes
REMOTES = {
    "habitat_1.5gb": "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip",
    "habitat_11gb": "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip",
}

DEFAULT_PATH = str(utils.get_project_root() / "resources/meshes")
if not os.path.exists(DEFAULT_PATH):
    os.makedirs(DEFAULT_PATH)

DEFAULT_CLEANUP = False
DEFAULT_REMOTE = "habitat_1.5gb"


def get_user_confirmation() -> None:
    """
    Prompts user to confirm they have signed the agreement for downloading the Gibson dataset
    """
    u = input(
        f"By running this script, you confirm that you have and uploaded a signed user agreement form to access the Gibson Environment database.\n"
        f"If you have not completed this, please go to {USER_AGREEMENT}, download the PDF, sign it, and upload it to the form.\n\n"
        f"Please type 'Y' to confirm that you have signed the user agreement."
    )
    if u.upper() != "Y":
        sys.exit(0)


def main(
    path: str = DEFAULT_PATH,
    cleanup: bool = DEFAULT_CLEANUP,
    remote: list[str] = DEFAULT_REMOTE,
) -> None:
    f"""
    Downloads and prepares the Gibson Environment dataset meshes.

    NOTE: by running this script, we assume that you have completed the user agreement form
    (https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform).

    Arguments:
        path: Path to store and process the dataset, defaults to {DEFAULT_PATH}.
        cleanup: Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}.
        remote: List of remote URLs to download from. Defaults to {DEFAULT_REMOTE}.
            Each value must be one of {", ".join(list(REMOTES.keys()))}
    """
    print("---- Gibson Environment Database download script ----")
    get_user_confirmation()
    print(f"Datasets will be downloaded to: {path}")
    print(f"Using remotes: {', '.join(remote)}")

    path = Path(path)

    for rem in remote:
        print(f"Downloading {rem}...")
        if rem not in REMOTES.keys():
            raise ValueError(
                f"Expected `remote` to be one of {', '.join(list(REMOTES.keys()))}, but got {rem}"
            )

        desired_remote = REMOTES[rem]

        zip_path = path / f"{rem}.zip"
        download_file(desired_remote, zip_path)
        print(f"... downloaded {rem} to {zip_path}")

        extract_zip(zip_path, path)
        print(f"... extracted {rem} to {path}")

        if cleanup:
            os.remove(zip_path)
            print(f"... removed {zip_path}")

    # Remove .navmesh files, we don't need them
    if cleanup:
        for f in (path / "gibson").glob("*.navmesh"):
            os.remove(path / f)


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
    parser.add_argument(
        "--remote",
        nargs="*",
        default=[DEFAULT_REMOTE],
        help=f"Remote files to download: defaults to {DEFAULT_REMOTE}. "
        f"Provide multiple values to download multiple datasets, e.g. --remote aaa --remote bbb --remote ccc. "
        f'Each value must be one of {", ".join(list(REMOTES.keys()))}',
    )

    args = vars(parser.parse_args())

    main(**args)
