#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates similar training data to that contained in [this repo](https://zenodo.org/records/6406873):

- 1200 1-minute long spatial recordings
- Sampling rate of 24kHz
- Two 4-channel recording formats, first-order Ambisonics (FOA) and tetrahedral microphone array (MIC)
    - Currently, only MIC is implemented
- Spatial events spatialized in 9 unique rooms, using measured RIRs for the two formats
    - 6 rooms designated as "training": 150 soundscapes created for each
    - 3 rooms designated as "testing": 100 soundscapes created for each
- Maximum polyphony of 2 (with possible same-class events overlapping)

One augmentation added to every audio file, sampled from:
- Pitch shifting (+/- up to half an octave)
- Time stretching (between 0.9 and 1.1x)
- Distortion (up to +10 dB gain)
- Reverse
- Phase inversion
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

# For reproducible randomisation
utils.seed_everything(utils.SEED)

# Filepaths, directories, etc.
FG_DIR = utils.get_project_root() / "resources/soundevents"
MESH_DIR = utils.get_project_root() / "resources/meshes/gibson"
OUTPUT_DIR = utils.get_project_root() / "spatial_scenes_dcase_synthetic"

# Data splits
TRAIN_RECORDINGS_PER_ROOM, TEST_RECORDINGS_PER_ROOM = 150, 100
# Randomly sampled, but hardcoded so they are the same across runs
TRAIN_ROOMS = [
    "Haymarket.glb",
    "Swisshome.glb",
    "Siren.glb",
    "Traver.glb",
    "Hercules.glb",
    "Halfway.glb",
]
TEST_ROOMS = ["Helix.glb", "Peacock.glb", "Vails.glb"]


# Parameters taken from DCASE data
DURATION = 60
SAMPLE_RATE = 24000
MAX_OVERLAP = 2

# Distributions to sample for events
STATIC_EVENTS = utils.sanitise_distribution(
    lambda: random.choice(range(config.MIN_STATIC_EVENTS, config.MAX_STATIC_EVENTS))
)
MOVING_EVENTS = utils.sanitise_distribution(
    lambda: random.choice(range(config.MIN_MOVING_EVENTS, config.MAX_MOVING_EVENTS))
)

# Valid materials for the ray-tracing engine
with open(MATERIALS_JSON, "r") as js_in:
    js_out = json.load(js_in)
VALID_MATERIALS = list({mat["name"] for mat in js_out["materials"]})


def generate(
    mesh_name: str, split: str, scene_num: int, scape_num: int, output_dir: Path
) -> None:
    """
    Make a single generation with required arguments
    """
    # Resolve full mesh_path
    mesh_path = MESH_DIR / mesh_name

    # Output filepaths
    fold = 1 if split == "train" else 2
    common = f"dev-{split}-alight/fold{fold}_scene{scene_num}_{str(scape_num).zfill(3)}"
    audio_path = output_dir / f"mic_dev/{common}.wav"
    metadata_path = output_dir / f"metadata_dev/{common}.csv"

    # Skip over this generation if files already exist
    if (
        audio_path.with_name(audio_path.stem + "_mic.wav").exists()
        and metadata_path.with_name(metadata_path.stem + "_mic.csv").exists()
    ):
        return

    # Choose a material to use
    scene_material = random.choice(VALID_MATERIALS)

    # Choose a noise floor for the scene
    scene_ref_db = np.random.uniform(config.MIN_REF_DB, config.MAX_REF_DB)

    scene = Scene(
        duration=DURATION,
        mesh_path=Path(mesh_path),
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
        event_augmentations=[
            Reverse,
            Invert,
            (PitchShift, dict(sample_rate=SAMPLE_RATE, semitones=stats.uniform(-7, 0))),
            (
                SpeedUp,
                dict(sample_rate=SAMPLE_RATE, stretch_factor=stats.uniform(0.9, 0.2)),
            ),
            (
                Distortion,
                dict(sample_rate=SAMPLE_RATE, drive_db=stats.uniform(0.0, 10.0)),
            ),
        ],
        fg_path=Path(FG_DIR),
        max_overlap=MAX_OVERLAP,
        ref_db=scene_ref_db,
        state_kwargs=dict(
            add_to_context=False,
            material=scene_material,
            rlr_kwargs=dict(sample_rate=SAMPLE_RATE),
        ),
        allow_duplicate_audios=False,
    )

    # Add the microphone, static + moving events (one augmentation sampled randomly from above list)
    #  skip over any errors when adding the event and just continue to the next one
    scene.add_microphone(microphone_type="foalistener", alias="mic")
    for _ in range(1):
        try:
            scene.add_event(
                event_type="static",
                augmentations=1,
                ensure_direct_path=True,
                max_place_attempts=100,
            )
        except ValueError as e:
            logger.warning(e)

    # for _ in range(MOVING_EVENTS.rvs()):
    #     try:
    #         scene.add_event(
    #             event_type="moving",
    #             augmentations=1,
    #             ensure_direct_path=True,
    #             max_place_attempts=100,
    #         )
    #     except ValueError as e:
    #         logger.warning(e)

    # Always add gaussian noise
    scene.add_ambience(noise="gaussian")

    # If no events added successfully, try again
    if len(scene.get_events()) == 0:
        generate(
            mesh_name,
            split=split,
            scene_num=scene_num,
            scape_num=scape_num,
            output_dir=output_dir,
        )

    # Do the generation: create audio and DCASE metadata
    scene.generate(
        audio_fname=audio_path,
        metadata_fname=metadata_path,
        audio=True,
        metadata_json=True,
        metadata_dcase=True,
    )

    # Also dump an image of the state
    fig = scene.state.create_plot()
    fig.savefig(metadata_path.with_suffix(".png").as_posix())


def main(outdir: str):
    # Create the output folders if they don't currently exist
    outdir = Path(outdir)
    for fp in [
        outdir / "metadata_dev/dev-train-alight",
        outdir / "metadata_dev/dev-test-alight",
        outdir / "mic_dev/dev-train-alight",
        outdir / "mic_dev/dev-test-alight",
    ]:
        if not fp.exists():
            os.makedirs(fp)

    # Start iterating to create the required number of training scenes
    logger.info("Generating training scenes...")
    full_start = time()
    for train_room_idx, train_room in enumerate(TRAIN_ROOMS):
        for train_scape_idx in tqdm(
            range(TRAIN_RECORDINGS_PER_ROOM),
            desc=f"Generating for train room {train_room_idx}...",
        ):
            generate(train_room, "train", train_room_idx, train_scape_idx, outdir)

    logger.info("Generating testing scenes...")
    for test_room_idx, test_room in enumerate(TEST_ROOMS):
        for test_scape_idx in tqdm(
            range(TEST_RECORDINGS_PER_ROOM),
            desc=f"Generating for test room {test_room_idx}...",
        ):
            generate(test_room, "test", test_room_idx, test_scape_idx, outdir)

    # Log the time taken
    full_end = time() - full_start
    logger.info(f"Finished in {full_end:.4f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates similar synthetic data to https://zenodo.org/records/6406873 using AudibleLight"
    )
    parser.add_argument(
        "--outdir",
        type=int,
        default=OUTPUT_DIR,
        help=f"Path to save generated outputs, defaults to {OUTPUT_DIR}",
    )
    args = vars(parser.parse_args())

    main(**args)
