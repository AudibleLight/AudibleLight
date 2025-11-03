#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates an example dataset for SELD, similar to DCASE 2023 Task 3 and [this repo](https://zenodo.org/records/6406873):

By default, this script can generate:
- 1200 1-minute long spatial recordings
- Sampling rate of 24kHz
- Two 4-channel recording formats, first-order Ambisonics (FOA) and tetrahedral microphone array (MIC)
- Spatial events spatialized in N unique spaces, using measured or ray-traced RIRs for the two formats
- Maximum polyphony of 2 (with possible same-class events overlapping)

A single augmentation can be to every audio file, sampled randomly from:
- Pitch shifting (+/- up to half an octave)
- Time stretching (between 0.9 and 1.1x)
- Distortion (up to +10 dB gain)
- Reverse
- Phase inversion

The backend can be controlled using the "--backend" flag, and defaults to RLR (ray-tracing). When using this backend,
materials can also be added to the simulation.

Many other parameters can be controlled using flags passed in to the script.
"""

import argparse
import json
import os
import random
from pathlib import Path
from time import time

import numpy as np
from loguru import logger
from scipy import stats
from tqdm import tqdm

from audiblelight import config, utils
from audiblelight.augmentation import Distortion, Invert, PitchShift, Reverse, SpeedUp
from audiblelight.core import Scene
from audiblelight.worldstate import MATERIALS_JSON
from scripts.seld.seld_dataset_assets import MESHES, SOFAS

# For reproducible randomisation
utils.seed_everything(utils.SEED)

# Filepaths, directories, etc.
FG_DIR = utils.get_project_root() / "resources/soundevents"
MESH_DIR = utils.get_project_root() / "resources/meshes/gibson"
SOFA_DIR = utils.get_project_root() / "resources/sofa/rirs"
OUTPUT_DIR = utils.get_project_root() / "spatial_scenes_dcase_synthetic"

# Parameters taken from DCASE data
DURATION = 60
SAMPLE_RATE = 24000

# Valid materials for the ray-tracing engine
with open(MATERIALS_JSON, "r") as js_in:
    js_out = json.load(js_in)
VALID_MATERIALS = list({mat["name"] for mat in js_out["materials"]})

AUGMENTATIONS = {
    "pitchshift": (
        PitchShift,
        dict(sample_rate=SAMPLE_RATE, semitones=stats.uniform(-7, 0)),
    ),
    "speedup": (
        SpeedUp,
        dict(sample_rate=SAMPLE_RATE, stretch_factor=stats.uniform(0.9, 0.2)),
    ),
    "reverse": Reverse,
    "invert": Invert,
    "distortion": (
        Distortion,
        dict(sample_rate=SAMPLE_RATE, drive_db=stats.uniform(0.0, 10.0)),
    ),
}


def get_augmentations(augmentation_names: list[str]) -> list:
    """
    Given a list of augmentation names as strings, grab them from the AUGMENTATIONS dictionary
    """
    grabbed = []
    for aug in augmentation_names:
        if aug in AUGMENTATIONS.keys():
            grabbed.append(AUGMENTATIONS[aug])
        else:
            raise ValueError(
                f"Augmentation {aug} is not a valid parameter for this script!"
            )
    return grabbed


def generate(
    backend: str,
    asset_name: str,
    split: str,
    scene_num: int,
    scape_num: int,
    output_dir: Path,
    channel_layout: str,
    augmentations: list[str],
    materials: bool,
    max_overlap: int,
    min_events_static,
    max_events_static: int,
    min_events_moving: int,
    max_events_moving: int,
) -> None:
    """
    Make a single generation with required arguments
    """
    # Output filepaths
    fold = 1 if split == "train" else 2
    common = f"dev-{split}-alight/fold{fold}_scene{scene_num}_{str(scape_num).zfill(3)}"
    audio_path = output_dir / f"{channel_layout}_dev/{common}.wav"
    metadata_path = output_dir / f"metadata_dev/{common}.csv"

    # Skip over this generation if files already exist
    if (
        audio_path.with_name(audio_path.stem + f"_{channel_layout}.wav").exists()
        and metadata_path.with_name(
            metadata_path.stem + f"_{channel_layout}.csv"
        ).exists()
    ):
        return None

    # Choose a noise floor for the scene
    scene_ref_db = np.random.uniform(config.MIN_REF_DB, config.MAX_REF_DB)

    # Use augmentations
    if augmentations:
        use_augmentations = get_augmentations(augmentations)
    else:
        use_augmentations = None

    # Distributions to sample for events
    static_events = utils.sanitise_distribution(
        lambda: random.choice(range(min_events_static, max_events_static + 1)),
    )
    moving_events = utils.sanitise_distribution(
        lambda: random.choice(range(min_events_moving, max_events_moving + 1)),
    )

    # Resolve full kwargs for backend
    if backend == "rlr":
        backend_kwargs = dict(
            add_to_context=False,
            material=random.choice(VALID_MATERIALS) if materials else "Default",
            mesh=MESH_DIR / asset_name,
        )

    elif backend == "sofa":
        # SOFA name also includes channel layout
        backend_kwargs = dict(
            sofa=SOFA_DIR / (asset_name + f"_{channel_layout}.sofa"),
        )
    else:
        raise ValueError("Unknown backend: '{}'".format(backend))

    # Initialise the Scene with all arguments
    scene = Scene(
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        backend=backend,
        scene_start_dist=stats.uniform(0.0, DURATION - 1),
        # Audio files will always start from 0 seconds in
        event_start_dist=None,
        # Events capped to 10 seconds
        event_duration_dist=stats.uniform(
            config.MIN_EVENT_DURATION,
            config.MAX_EVENT_DURATION - config.MIN_EVENT_DURATION,
        ),
        # Events have speed between 0.5 and 2.0 metres-per-second
        event_velocity_dist=stats.uniform(
            config.MIN_EVENT_VELOCITY,
            config.MAX_EVENT_VELOCITY - config.MIN_EVENT_VELOCITY,
        ),
        # Events have resolution between 1.0 and 4.0 Hz
        event_resolution_dist=stats.uniform(
            config.MIN_EVENT_RESOLUTION,
            config.MAX_EVENT_RESOLUTION - config.MIN_EVENT_RESOLUTION,
        ),
        # Events have SNR between 5 and 30 dB
        snr_dist=stats.uniform(
            config.MIN_EVENT_SNR, config.MAX_EVENT_SNR - config.MIN_EVENT_SNR
        ),
        # Event augmentations will sample from this list
        event_augmentations=use_augmentations,
        fg_path=Path(FG_DIR),
        max_overlap=max_overlap,
        ref_db=scene_ref_db,
        backend_kwargs=backend_kwargs,
        allow_duplicate_audios=False,
    )

    # Add the microphone to ray-tracing/parameterized backends
    if backend != "sofa":
        scene.add_microphone(
            microphone_type="ambeovr" if channel_layout == "mic" else "foalistener",
            alias=channel_layout,
        )

    # Add static + moving events (one augmentation sampled randomly from above list)
    # skip over any errors when adding the event and just continue to the next one
    for _ in range(static_events.rvs()):
        try:
            scene.add_event(
                event_type="static",
                augmentations=1 if use_augmentations else None,
                ensure_direct_path=True,
                max_place_attempts=100,
            )
        except ValueError as e:
            logger.warning(e)

    for _ in range(moving_events.rvs()):
        # Sample the shape to use for this moving event: one of random walk, semicircular, linear
        shape = random.choice(config.MOVING_EVENT_SHAPES)
        try:
            scene.add_event(
                event_type="moving",
                augmentations=1 if use_augmentations else None,
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
        return generate(
            backend=backend,
            asset_name=asset_name,
            split=split,
            scene_num=scene_num,
            scape_num=scape_num,
            output_dir=output_dir,
            channel_layout=channel_layout,
            augmentations=augmentations,
            materials=materials,
            max_overlap=max_overlap,
            min_events_static=min_events_static,
            min_events_moving=min_events_moving,
            max_events_static=max_events_static,
            max_events_moving=max_events_moving,
        )

    # Do the generation: create audio and DCASE metadata
    else:
        scene.generate(
            audio_fname=audio_path,
            metadata_fname=metadata_path,
            audio=True,
            metadata_json=True,
            metadata_dcase=True,
        )

        return None


def get_assets(backend: str, asset_split: str) -> dict:
    """
    Get the train + test meshes for this backend and split
    """
    if backend == "rlr":
        if str(asset_split) not in MESHES.keys():
            raise ValueError(
                f"Expected meshes in {list(MESHES.keys())} but got {asset_split}"
            )
        return MESHES[str(asset_split)]

    elif backend == "sofa":
        if str(asset_split) not in SOFAS.keys():
            raise ValueError(
                f"Expected .sofa files in {list(SOFAS.keys())} but got {asset_split}"
            )
        return SOFAS[str(asset_split)]

    else:
        raise ValueError(f"Unknown backend {backend}")


def main(
    backend: str,
    channel_layout: str,
    augmentations: bool,
    materials: bool,
    assets: str,
    outdir: str,
    max_overlap: int,
    min_events_static,
    max_events_static: int,
    min_events_moving: int,
    max_events_moving: int,
):
    """
    Runs the generation across all training + test rooms
    """
    # Parse the channel layout and microphone type
    if channel_layout not in ["mic", "foa"]:
        raise ValueError(
            "Expected channel_layout 'mic' or 'foa' but got {}".format(channel_layout)
        )

    # Create the output folders if they don't currently exist
    outdir = Path(outdir)
    for fp in [
        outdir / "metadata_dev/dev-train-alight",
        outdir / "metadata_dev/dev-test-alight",
        outdir / f"{channel_layout}_dev/dev-train-alight",
        outdir / f"{channel_layout}_dev/dev-test-alight",
    ]:
        if not fp.exists():
            os.makedirs(fp)

    # Get the train + test meshes for this run
    chosen = get_assets(backend, assets)
    train_rooms = chosen["train"]
    test_rooms = chosen["test"]
    train_recordings_per_room = chosen["scapes_per_train_mesh"]
    test_recordings_per_room = chosen["scapes_per_test_mesh"]

    # Start iterating to create the required number of training scenes
    logger.info("Generating training scenes...")
    full_start = time()
    for train_room_idx, train_room in enumerate(train_rooms):
        for train_scape_idx in tqdm(
            range(train_recordings_per_room),
            desc=f"Generating for train room {train_room_idx + 1}/{len(train_rooms)}, name {train_room}...",
        ):
            generate(
                backend,
                train_room,
                "train",
                train_room_idx,
                train_scape_idx,
                outdir,
                channel_layout,
                augmentations,
                materials,
                max_overlap,
                min_events_static,
                max_events_static,
                min_events_moving,
                max_events_moving,
            )

    logger.info("Generating testing scenes...")
    for test_room_idx, test_room in enumerate(test_rooms):
        for test_scape_idx in tqdm(
            range(test_recordings_per_room),
            desc=f"Generating for test room {test_room_idx + 1}/{len(test_rooms)}, name {test_room}...",
        ):
            generate(
                backend,
                test_room,
                "test",
                test_room_idx,
                test_scape_idx,
                outdir,
                channel_layout,
                augmentations,
                materials,
                max_overlap,
                min_events_static,
                max_events_static,
                min_events_moving,
                max_events_moving,
            )

    # Log the time taken
    full_end = time() - full_start
    logger.info(f"Finished in {full_end:.4f} seconds.")


if __name__ == "__main__":
    # Use module docstring for the help text
    parser = argparse.ArgumentParser(description=__doc__)

    # Here come the user parameters
    parser.add_argument(
        "--backend",
        type=str,
        default=config.DEFAULT_BACKEND,
        help="The backend to use, defaults to 'rlr'.",
    )
    parser.add_argument(
        "--channel-layout",
        type=str,
        default=config.DEFAULT_CHANNEL_LAYOUT,
        help="The channel layout to use, must be either 'foa' or 'mic'",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        help=f"The name of augmentations to use: supported are {', '.join(AUGMENTATIONS.keys())}",
        default=None,
    )
    parser.add_argument(
        "--materials",
        action="store_true",
        help="Add this flag to use materials with 'backend=rlr'",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="9A",
        help="The data files (.glb meshes or .sofa) to use: see `seld_dataset_assets.py`. "
        "Note that the total number of scapes to generate will remain fixed at 1200.",
    )
    parser.add_argument(
        "--outdir",
        type=int,
        default=OUTPUT_DIR,
        help=f"Path to save generated outputs, defaults to {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=config.MAX_OVERLAP,
        help=f"Maximum number of overlapping events, defaults to {config.MAX_OVERLAP}",
    )
    parser.add_argument(
        "--min-events-static",
        type=int,
        default=config.MIN_STATIC_EVENTS,
        help=f"Minimum number of static events per scene, defaults to {config.MIN_STATIC_EVENTS}",
    )
    parser.add_argument(
        "--max-events-static",
        type=int,
        default=config.MAX_STATIC_EVENTS,
        help=f"Maximum number of static events per scene, defaults to {config.MAX_STATIC_EVENTS}",
    )
    parser.add_argument(
        "--min-events-moving",
        type=int,
        default=config.MIN_MOVING_EVENTS,
        help=f"Minimum number of moving events per scene, defaults to {config.MIN_MOVING_EVENTS}",
    )
    parser.add_argument(
        "--max-events-moving",
        type=int,
        default=config.MAX_MOVING_EVENTS,
        help=f"Maximum number of static events per scene, defaults to {config.MAX_MOVING_EVENTS}",
    )

    # Parse args and start generating
    args = vars(parser.parse_args())
    logger.info("Generating with args: {}".format(args))

    main(**args)
