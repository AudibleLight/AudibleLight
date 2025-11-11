#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and prepares the Free Music Archive."""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from audiblelight.utils import get_project_root, sanitise_positive_number
from scripts.download_data.utils import BaseDataSetup, download_file, extract_zip

BASE_URL = "https://os.unil.cloud.switch.ch/fma/"
METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

REMOTES = {
    "fma_small": BASE_URL + "fma_small.zip",
    "fma_medium": BASE_URL + "fma_medium.zip",
    "fma_large": BASE_URL + "fma_large.zip",
    "fma_full": BASE_URL + "fma_full.zip",
}
CORRUPT_FMA_TRACKS = ["098565", "098567", "098569", "099134", "108925", "133297"]
SKIP_GENRES = ["Electronic", "Experimental", "Instrumental"]
DCASE_FSD50K_SELECTED = "https://zenodo.org/record/6406873/files/FSD50K_selected.txt"

DEFAULT_PATH = str(get_project_root() / "resources/soundevents")
DEFAULT_CLEANUP = False
DEFAULT_REMOTE = ["fma_small"]
DEFAULT_NTRACKS = 30


class FMADataSetup(BaseDataSetup):
    def __init__(
        self,
        dataset_name: str = "fma_small",
        ntracks_genre: int = 20,
        split_prob: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ntracks_genre = sanitise_positive_number(ntracks_genre, cast_to=int)
        self.split_prob = sanitise_positive_number(split_prob, cast_to=float)
        self.dataset_name = dataset_name

        if self.dataset_name not in REMOTES:
            raise ValueError(
                f"Expected 'dataset_name' to be one of {', '.join(list(REMOTES.keys()))}, but got {dataset_name}"
            )

        self.zip_name = self.dataset_name + ".zip"
        self.base_url = REMOTES[self.dataset_name]
        self.metadata_url = METADATA_URL

    def prepare_dataset(self) -> None:
        """
        Prepares the FMA dataset by downloading and extracting it.
        """
        dataset_path = os.path.join(self.dataset_home, self.dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Downloading {self.dataset_name} dataset...")
            self.download_dataset()
        else:
            print(f"{self.dataset_name} dataset already exists. Skipping download.")
        self.gen_dataset_splits()

    def download_dataset(self) -> None:
        """
        Downloads and extracts the FMA dataset.
        """
        os.makedirs(self.dataset_home, exist_ok=True)
        download_file(self.base_url, os.path.join(self.dataset_home, self.zip_name))
        extract_zip(os.path.join(self.dataset_home, self.zip_name), self.dataset_home)
        print("Done downloading and unzipping FMA")
        download_file(
            self.metadata_url, os.path.join(self.dataset_home, "meta_" + self.zip_name)
        )
        extract_zip(
            os.path.join(self.dataset_home, "meta_" + self.zip_name), self.dataset_home
        )
        print("Done downloading and unzipping metadata")

    def gen_dataset_splits(self, target_subdir: str = DEFAULT_PATH) -> None:
        """
        Generates dataset splits for training and testing.
        """
        path_to_fsd50k_dcase = os.path.join(self.dataset_home, target_subdir)
        tracks = pd.read_csv(
            os.path.join(self.dataset_home, "fma_metadata/tracks.csv"),
            header=[0, 1],
            index_col=0,
        )
        genres = tracks["track"]["genre_top"].unique()

        for genre in genres:

            if genre in SKIP_GENRES or pd.isna(genre):
                continue

            genre_tracks = tracks[
                (tracks["track", "genre_top"] == genre)
                & (tracks["set", "subset"] == self.dataset_name.split("_")[1])
            ]
            selected_tracks = genre_tracks[: self.ntracks_genre]

            if not genre_tracks.empty:
                genre_dir_train = os.path.join(
                    path_to_fsd50k_dcase, "music", "train", genre
                )
                genre_dir_test = os.path.join(
                    path_to_fsd50k_dcase, "music", "test", genre
                )
                os.makedirs(genre_dir_train, exist_ok=True)
                os.makedirs(genre_dir_test, exist_ok=True)

            for track_id, track in selected_tracks.iterrows():
                if str(track_id).zfill(6) in CORRUPT_FMA_TRACKS:
                    continue
                fma_track_path = os.path.join(
                    self.dataset_home,
                    self.dataset_name,
                    str(track_id).zfill(6)[:3],
                    f"{track_id:06}.mp3",
                )
                fold = "train" if np.random.rand() < self.split_prob else "test"
                dcase_path = os.path.join(
                    path_to_fsd50k_dcase, "music", fold, genre, f"{track_id:06}.mp3"
                )
                shutil.copyfile(fma_track_path, dcase_path)


def main(
    path: str = DEFAULT_PATH,
    cleanup: bool = DEFAULT_CLEANUP,
    remote: list[str] = DEFAULT_REMOTE,
    ntracks: int = DEFAULT_NTRACKS,
):
    f"""
    Downloads and prepares the Free Music Archive.

    Arguments:
        path: Path to store and process the dataset, defaults to {DEFAULT_PATH}.
        cleanup: Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}.
        remote: List of remote URLs to download from. Defaults to {DEFAULT_REMOTE}.
            Each value must be one of {", ".join(list(REMOTES.keys()))}
        ntracks: keep this many tracks per genre, defaults to {DEFAULT_NTRACKS}.
    """
    print("---- Free Music Archive download script ----")
    print(f"Datasets will be downloaded to: {path}")
    print(f"Using remotes: {', '.join(remote)}")

    for rem in remote:
        fma = FMADataSetup(dataset_name=rem, dataset_home=path, ntracks_genre=ntracks)
        fma.prepare_dataset()
        if cleanup:
            fma.cleanup()

            # Remove additional folders we've created
            #  Try block is needed as we might have removed these inside `cleanup` method
            for f in [rem, "fma_metadata"]:
                try:
                    shutil.rmtree(os.path.join(path, f))
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
    parser.add_argument(
        "--remote",
        nargs="*",
        default=DEFAULT_REMOTE,
        help=f"Remote files to download: defaults to {DEFAULT_REMOTE}. "
        f"Provide multiple values to download multiple datasets, e.g. --remote aaa --remote bbb --remote ccc. "
        f'Each value must be one of {", ".join(list(REMOTES.keys()))}',
    )
    parser.add_argument(
        "--ntracks",
        default=DEFAULT_NTRACKS,
        help=f"Keep this many tracks per genre, defaults to {DEFAULT_NTRACKS}",
    )
    args = vars(parser.parse_args())
    main(**args)
