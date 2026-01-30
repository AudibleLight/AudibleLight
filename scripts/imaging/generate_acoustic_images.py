#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file generates acoustic images and associated labels using AudibleLight.

The acoustic images comprise HDF files, which contain arrays with shape (tesselation, n_bands, n_frames), and also JSON
files, which contain the locations of polygons for sound events within the Scene. These JSON files contain the pixel
coordinates and amplitude values for the acoustic images, which are standardised using the distribution of values
found in the actual STARSS23 dataset.
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from loguru import logger
from scipy import stats

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.worldstate import WorldStateRLR

# For reproducible randomisation
utils.seed_everything(utils.SEED)

# Acoustic image configuration
#  Default values taken from LAM paper
AIMG_FMIN, AIMG_FMAX = 1500, 4500
AIMG_NBANDS = 9
AIMG_SCALE = "linear"
AIMG_BANDWIDTH = 50.0
AIMG_TSTI = 10e-3
AIMG_FRAME_CAP = None
AIMG_SH_ORDER = 10
AIMG_CIRCLE_RADIUS_DEG = 20
AIMG_POLYGON_MASK_THRESHOLD = 4e-5
AIMG_RESOLUTION = 640, 320

# Filepaths, directories, etc.
FG_DIR = utils.get_project_root() / "resources/soundevents"
MESH_DIR = utils.get_project_root() / "resources/meshes/gibson"

# Output directory: change this to wherever you want
OUTPUT_DIR = utils.get_project_root() / "acoustic_images"

# Scene qualities
DURATION = 10
SAMPLE_RATE = 24000
BACKEND = "rlr"
MIC_TYPE = "eigenmike32"
REF_DB = -65  # noise floor

# Event qualities + number
MAX_OVERLAP = 2
MIN_STATIC_EVENTS, MAX_STATIC_EVENTS = 1, 10
MIN_MOVING_EVENTS, MAX_MOVING_EVENTS = 0, 6
MIN_EVENT_DURATION, MAX_EVENT_DURATION = 2.0, 10.0
MIN_EVENT_VELOCITY, MAX_EVENT_VELOCITY = 0.5, 2.0
MIN_EVENT_RESOLUTION, MAX_EVENT_RESOLUTION = 1.0, 4.0
MIN_EVENT_SNR, MAX_EVENT_SNR = 5, 30
MOVING_EVENT_SHAPES = ["random", "linear", "semicircular"]

# Data splits
TRAIN_SIZE, VALID_SIZE = 0.8, 0.2
N_SCENES = 1000
#  Get all meshes and shuffle them up
ALL_MESHES = list(MESH_DIR.rglob("**/*.glb"))
random.shuffle(ALL_MESHES)
#  Now, get the required number of meshes for each split
TRAIN_N_MESHES = round(N_SCENES * TRAIN_SIZE)
VALID_N_MESHES = round(N_SCENES * VALID_SIZE)
assert TRAIN_N_MESHES + VALID_N_MESHES == N_SCENES
#  Now, partition the meshes
TRAIN_MESHES = ALL_MESHES[:TRAIN_N_MESHES]
VALID_MESHES = ALL_MESHES[TRAIN_N_MESHES:]


def generate(
    mesh_path: Union[str, Path],
    output_dir: Union[str, Path],
    split: str,
    scape_num: int,
):
    # Setting up output filepaths
    fold = 1 if split == "train" else 2
    common = f"dev-{split}-alight/fold{fold}_scape{str(scape_num).zfill(5)}"
    audio_path = output_dir / f"mic_dev/{common}.wav"
    aimg_path = output_dir / f"aimg_dev/{common}.hdf"
    dcase_labels_path = output_dir / f"metadata_dev/{common}.csv"
    aimg_labels_path = output_dir / f"aimg_labels_dev/{common}.json"

    # Create the scene with all the default arguments and the given mesh
    scene = Scene(
        duration=DURATION,
        backend=WorldStateRLR(
            mesh=mesh_path,
            sample_rate=SAMPLE_RATE,
            add_to_context=False,
            material="Default",
        ),
        sample_rate=SAMPLE_RATE,
        fg_path=FG_DIR,
        max_overlap=MAX_OVERLAP,
        ref_db=REF_DB,
        allow_duplicate_audios=False,
        event_augmentations=None,
        scene_start_dist=stats.uniform(0.0, DURATION - 1),
        # Audio files will always start from 0 seconds in
        event_start_dist=None,
        # Events capped to 10 seconds
        event_duration_dist=stats.uniform(
            MIN_EVENT_DURATION,
            MAX_EVENT_DURATION - MIN_EVENT_DURATION,
        ),
        # Events have speed between 0.5 and 2.0 metres-per-second
        event_velocity_dist=stats.uniform(
            MIN_EVENT_VELOCITY,
            MAX_EVENT_VELOCITY - MIN_EVENT_VELOCITY,
        ),
        # Events have resolution between 1.0 and 4.0 Hz
        event_resolution_dist=stats.uniform(
            MIN_EVENT_RESOLUTION,
            MAX_EVENT_RESOLUTION - MIN_EVENT_RESOLUTION,
        ),
        # Events have SNR between 5 and 30 dB
        snr_dist=stats.uniform(MIN_EVENT_SNR, MAX_EVENT_SNR - MIN_EVENT_SNR),
    )

    # Add the eigenmike32 in
    scene.add_microphone(microphone_type=MIC_TYPE, alias="em32")

    # Distributions to sample for events
    static_events = utils.sanitise_distribution(
        lambda: random.choice(range(MIN_STATIC_EVENTS, MAX_STATIC_EVENTS + 1)),
    )
    moving_events = utils.sanitise_distribution(
        lambda: random.choice(range(MIN_MOVING_EVENTS, MAX_MOVING_EVENTS + 1)),
    )

    # Add static + moving events
    #  skip over any errors when adding the event and just continue to the next one
    for _ in range(static_events.rvs()):
        try:
            scene.add_event(
                event_type="static",
                augmentations=None,
                ensure_direct_path=True,
                max_place_attempts=100,
            )
        except ValueError as e:
            logger.warning(e)

    for _ in range(moving_events.rvs()):
        # Sample the shape to use for this moving event: one of random walk, semicircular, linear
        shape = random.choice(MOVING_EVENT_SHAPES)
        try:
            scene.add_event(
                event_type="moving",
                augmentations=None,
                ensure_direct_path=True,
                max_place_attempts=100,
                shape=shape,
            )
        except ValueError as e:
            logger.warning(e)

    # Always add gaussian noise
    scene.add_ambience(noise="gaussian")

    # If no events added successfully, try again by calling the function recursively
    if len(scene.get_events()) == 0:
        return generate(mesh_path, output_dir, split, scape_num)

    else:
        # Generate the audio and the acoustic image
        scene.generate(
            audio=True,
            metadata_json=True,
            metadata_dcase=False,
            audio_fname=audio_path,
            metadata_fname=dcase_labels_path,
        )
        scene.generate_acoustic_image(
            json_fname=aimg_labels_path,
            hdf_fname=aimg_path,
            frame_cap=AIMG_FRAME_CAP,
            nbands=AIMG_NBANDS,
            scale=AIMG_SCALE,
        )

        # Added to satisfy type checking
        return None


def main(output_dir: Union[str, Path]) -> None:
    # Check output directory exists, make if not
    output_dir = utils.sanitise_directory(output_dir, create_if_missing=True)

    # Create the output folders if they don't currently exist
    for fp in [
        output_dir / "aimg_dev/dev-train-alight",
        output_dir / "aimg_dev/dev-test-alight",
        output_dir / "metadata_dev/dev-train-alight",
        output_dir / "metadata_dev/dev-test-alight",
        output_dir / "aimg_labels_dev/dev-train-alight",
        output_dir / "aimg_labels_dev/dev-test-alight",
        output_dir / "mic_dev/dev-train-alight",
        output_dir / "mic_dev/dev-test-alight",
    ]:
        if not fp.exists():
            os.makedirs(fp)

    # Start generating the training scenes
    logger.info("Generating training acoustic images...")
    for train_room_idx in range(TRAIN_N_MESHES):
        # Grab a random train room
        train_mesh_path = random.choice(TRAIN_MESHES)
        generate(train_mesh_path, output_dir, "train", train_room_idx)

    # Start generating the testing scenes
    logger.info("Generating validation acoustic images...")
    for valid_room_idx in range(VALID_N_MESHES):
        # Grab a random valid room
        valid_mesh_path = random.choice(VALID_MESHES)
        generate(valid_mesh_path, output_dir, "test", valid_room_idx)


if __name__ == "__main__":
    # Use module docstring for the help text
    parser = ArgumentParser(description=__doc__)

    # Here come the user parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="The output directory, defaults to ./acoustic_images from repository root",
    )
    args = vars(parser.parse_args())
    main(**args)
