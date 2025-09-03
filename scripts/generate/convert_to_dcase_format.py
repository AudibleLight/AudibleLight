#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert multiple scenes to the format expected by [this repo](https://github.com/sharathadavanne/seld-dcase2023)"""

import argparse
import random
import shutil
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from audiblelight import utils

INPUT_DIR = utils.get_project_root() / "spatial_scenes"
OUTPUT_DIR = utils.get_project_root() / "seld-dcase2023-data"

DCASE_SAMPLE_RATE = 24000
TRAIN_SIZE = 0.8


def get_valid_scenes(input_dir: Path) -> list[Path]:
    """
    Return a list of valid directories, one per scene.
    """

    def _validate(dirpath: Path):
        """
        To be valid, folder must be a directory with both WAV and CSV files inside.
        """
        return all(
            (
                dirpath.is_dir(),
                len(list(dirpath.glob("*.wav"))) == 1,
                len(list(dirpath.glob("*.csv"))) == 1,
            )
        )

    return [i for i in input_dir.glob("*") if _validate(i)]


def split_scenes(
    scene_list: list[Path], train_size: float
) -> tuple[list[Path], list[Path]]:
    """
    Split scenes into train and validation sets.
    """
    # Get index needed to do the splitting
    split_idx = round(train_size * len(scene_list))

    # Shuffle the list of scenes up
    random.shuffle(scene_list)

    # Do the splitting
    train_scenes, val_scenes = scene_list[:split_idx], scene_list[split_idx:]

    # Sanity check: should be no overlap, and all scenes should be used
    assert len(list(set(train_scenes) & set(val_scenes))) == 0
    assert len(train_scenes) + len(val_scenes) == len(scene_list)

    return train_scenes, val_scenes


def copy_files(scene: Path, audio_dir: Path, meta_dir: Path, fold: int) -> None:
    """
    Copies the files (audio + metadata) from a scene to the desired directories
    """
    # Get the audio and metadata files within the folder
    audio = list(scene.glob("*.wav"))[0]
    meta = list(scene.glob("*.csv"))[0]

    # Get the name of the scene (name of the parent directory)
    scene_name = str(audio.parts[-2])

    # Save audio: needs to be multichannel with 24000 Hz sample rate
    audio_newpath = audio_dir / f"fold{fold}_{scene_name}.wav"
    y, _ = librosa.load(
        audio, sr=DCASE_SAMPLE_RATE, mono=False, offset=0.0, duration=None
    )
    n_channels, n_samples = y.shape
    if n_channels < 2:
        raise ValueError(
            f"Expected multichannel audio, got {n_channels} channels instead"
        )
    sf.write(audio_newpath, y.T, DCASE_SAMPLE_RATE)

    # Save metadata: should have 6 columns, no header, no index, as integers
    meta_newpath = meta_dir / f"fold{fold}_{scene_name}.csv"
    df = pd.read_csv(meta).astype(int)
    has_header = not all(isinstance(col, str) for col in df.columns)
    has_index = not len(df.columns) == 6
    df.to_csv(
        meta_newpath, sep=",", encoding="utf-8", header=has_header, index=has_index
    )


def process_split(split_dirs: list[Path], split_name: str, output_dir: Path) -> None:
    """
    Process all the files for one split
    """
    # Get fold number: DCASE-2023 expects folds 1-3 to be train, 4 to be test
    fold = 1 if split_name == "train" else 4

    # Sanitise output directories, create if not existing
    audio_dir = utils.sanitise_directory(
        output_dir / f"mic_dev/dev-{split_name}-alight", create_if_missing=True
    )
    meta_dir = utils.sanitise_directory(
        output_dir / f"metadata_dev/dev-{split_name}-alight", create_if_missing=True
    )

    # Iterate over all scenes and copy required files
    for scene in tqdm(split_dirs, desc=f"Processing files for split {split_name}"):
        copy_files(scene, audio_dir, meta_dir, fold)


def zip_and_tidy(dirpath: Path) -> None:
    """
    Archives everything in `dirpath` to a zip with the same name, then deletes `dirpath`.
    """
    shutil.make_archive(dirpath, "zip", dirpath)
    shutil.rmtree(dirpath)


def main(input_dir: str, output_dir: str, train_size: float) -> None:
    """
    Gets all AudibleLight outputs, splits into test/validation sets, copies all files, and dumps ZIP archives
    """
    # Sanitise dirpaths, create output directory of missing
    input_dir = utils.sanitise_directory(input_dir)
    output_dir = utils.sanitise_directory(output_dir, create_if_missing=True)
    logger.info(
        f"Searching for AudibleLight outputs inside {str(input_dir.resolve())}..."
    )

    # Sanitise split size
    train_size = utils.sanitise_positive_number(train_size)

    # Get a list of valid scene directories
    scenes = get_valid_scenes(input_dir)
    if len(scenes) == 0:
        raise ValueError(f"Could not find valid scenes inside {str(input_dir)}!")
    logger.info(f"Found {len(scenes)} scenes to use...")

    # Split valid scenes into train/test lists
    train_scenes, val_scenes = split_scenes(scenes, train_size)
    logger.info(
        f"Split into {len(train_scenes)} train scenes, {len(val_scenes)} validation scenes..."
    )

    # Coerce individual scenes into correct format for DCASE
    process_split(train_scenes, "train", output_dir)
    process_split(val_scenes, "test", output_dir)

    # Zip everything up and remove the directory
    zip_and_tidy(output_dir / "mic_dev")
    zip_and_tidy(output_dir / "metadata_dev")

    logger.info(
        f"Done! The audio and metadata zip files can be found in {str(output_dir.resolve())}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of AudibleLight metadata JSON files to DCASE format."
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, default=INPUT_DIR, help="Input directory"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--train-size", type=float, default=TRAIN_SIZE, help="Size of train split"
    )
    args = vars(parser.parse_args())
    main(**args)
