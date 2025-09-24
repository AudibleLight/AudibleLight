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
"""


import argparse
import os
import random
from pathlib import Path
from time import time

import pandas as pd
from loguru import logger
from scipy import stats
from tqdm import tqdm

from audiblelight import config, utils
from audiblelight.core import Scene

# For reproducible randomisation
utils.seed_everything(utils.SEED)

# Filepaths, directories, etc.
FG_DIR = utils.get_project_root() / "resources/soundevents"
MESH_DIR = utils.get_project_root() / "resources/meshes"
MESHES = list(MESH_DIR.rglob("*.glb"))
OUTPUT_DIR = utils.get_project_root() / "spatial_scenes_dcase_synthetic"

# Data splits
TRAIN_N_ROOMS, TEST_N_ROOMS = 6, 3
TRAIN_RECORDINGS_PER_ROOM, TEST_RECORDINGS_PER_ROOM = 150, 100
TRAIN_ROOMS = random.sample(MESHES, TRAIN_N_ROOMS)
TEST_ROOMS = random.sample([i for i in MESHES if i not in TRAIN_ROOMS], TEST_N_ROOMS)

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

# Types of noise we'll add
NOISE_TYPES = ["pink", "brown", "red", "blue", "white", "violet"]


def generate(
    mesh_path: str, split: str, scene_num: int, scape_num: int, output_dir: Path
) -> None:
    """
    Make a single generation with required arguments
    """
    # Output filepaths
    fold = 1 if split == "train" else 2
    common = f"dev-{split}-alight/fold{fold}_scene{scene_num}_{str(scape_num).zfill(3)}"
    audio_path = output_dir / f"mic_dev/{common}.wav"
    metadata_path = output_dir / f"metadata_dev/{common}.csv"

    # Skip over this generation if files already exist
    if audio_path.exists() and metadata_path.exists():
        return

    scene = Scene(
        duration=DURATION,
        mesh_path=Path(mesh_path),
        scene_start_dist=stats.uniform(0.0, DURATION - 1),
        event_start_dist=None,
        event_duration_dist=stats.uniform(
            config.MIN_EVENT_DURATION,
            config.MAX_EVENT_DURATION - config.MIN_EVENT_DURATION,
        ),
        event_velocity_dist=stats.uniform(
            config.MIN_EVENT_VELOCITY,
            config.MAX_EVENT_VELOCITY - config.MIN_EVENT_VELOCITY,
        ),
        event_resolution_dist=stats.uniform(
            config.MIN_EVENT_RESOLUTION,
            config.MAX_EVENT_RESOLUTION - config.MIN_EVENT_RESOLUTION,
        ),
        snr_dist=stats.uniform(
            config.MIN_EVENT_SNR, config.MAX_EVENT_SNR - config.MIN_EVENT_SNR
        ),
        fg_path=Path(FG_DIR),
        max_overlap=MAX_OVERLAP,
        ref_db=config.REF_DB,
        state_kwargs=dict(
            add_to_context=False, rlr_kwargs=dict(sample_rate=SAMPLE_RATE)
        ),
        allow_duplicate_audios=False,
    )

    # Add the microphone, static + moving events, and ambience
    scene.add_microphone(microphone_type=config.MIC_ARRAY_TYPE, alias="mic")
    for _ in range(STATIC_EVENTS.rvs()):
        scene.add_event(event_type="static")
    for _ in range(MOVING_EVENTS.rvs()):
        scene.add_event(event_type="moving")
    scene.add_ambience(noise=random.choice(NOISE_TYPES))

    # Do the generation: create audio and DCASE metadata
    scene.generate(
        audio_fname=audio_path,
        metadata_fname=metadata_path,
        audio=True,
        metadata_json=True,
        metadata_dcase=True,
    )


def dump_room_csv(outdir: Path) -> None:
    """
    Dumps a CSV file with the paths + splits of the rooms used
    """
    # List comprehensions -> list of dicts -> dataframe -> CSV
    room_paths = [{"split": "train", "mesh": str(p.resolve())} for p in TRAIN_ROOMS]
    room_paths.extend([{"split": "test", "mesh": str(p.resolve())} for p in TEST_ROOMS])
    room_df = pd.DataFrame(room_paths)
    room_df.to_csv(outdir / "rooms.csv", index=True)


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

    # Dump a CSV file containing the rooms + splits
    dump_room_csv(outdir)

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
