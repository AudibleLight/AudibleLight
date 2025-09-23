#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates similar training data to that reported in the SpatialScaper paper, Experiment 1.

- Chooses 11 random meshes
- Produces 150 1-minute scenes for each mesh (i.e., 1650 soundscapes)
    - Foreground sound events taken from FSD and FMA
    - Background sound events are different forms of noise (white/pink/red etc.)
- Models are trained on STARSS `dev-train` + DCASE synthetic data + AudibleLight synthetic data
- Models are validated on STARSS `dev-test-tau`
- Models are tested on STARSS `dev-test-japan`

STARSS and DCASE synthetic data should be prepared separately. The repositories are:
- [STARSS23](https://zenodo.org/records/7880637)
- [DCASE Synthetic SELD Mixtures for Baseline Training](https://zenodo.org/records/6406873)
"""


import argparse
import os
import random
from pathlib import Path
from time import time

from loguru import logger
from tqdm import tqdm

from audiblelight import config, utils
from scripts.experiments.dcase_synthetic_data import generate

FG_DIR = utils.get_project_root() / "resources/soundevents"
MESH_DIR = utils.get_project_root() / "resources/meshes"
MESHES = list(MESH_DIR.rglob("*.glb"))
OUTPUT_DIR = utils.get_project_root() / "spatial_scenes_sscaper_exp1"

# Data splits: only training here
TRAIN_N_ROOMS = 11
TRAIN_RECORDINGS_PER_ROOM = 150
TRAIN_ROOMS = random.sample(MESHES, TRAIN_N_ROOMS)

# Distributions to sample
STATIC_EVENTS = utils.sanitise_distribution(
    lambda: random.choice(range(config.MIN_STATIC_EVENTS, config.MAX_STATIC_EVENTS))
)
MOVING_EVENTS = utils.sanitise_distribution(
    lambda: random.choice(range(config.MIN_MOVING_EVENTS, config.MAX_MOVING_EVENTS))
)

# Types of noise we'll add
NOISE_TYPES = ["pink", "brown", "red", "blue", "white", "violet"]


def main(outdir: str):
    # Create the output folders if they don't currently exist
    outdir = Path(outdir)
    for fp in [
        outdir / "metadata_dev/dev-train-alight",
        outdir / "mic_dev/dev-train-alight",
    ]:
        if not fp.exists():
            os.makedirs(fp)

    # Start iterating to create the required number of training scenes
    logger.info("Generating scenes...")
    full_start = time()
    for train_room_idx, train_room in enumerate(TRAIN_ROOMS):
        for train_scape_idx in tqdm(
            range(TRAIN_RECORDINGS_PER_ROOM),
            desc=f"Generating for room {train_room_idx}...",
        ):
            generate(train_room, "train", train_room_idx, train_scape_idx, outdir)

    # Log the time taken
    full_end = time() - full_start
    logger.info(f"Finished in {full_end:.4f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates synthetic data equivalent to Experiment 1, SpatialScaper (Roman et al., 2024, ICASSP)"
    )
    parser.add_argument(
        "--outdir",
        type=int,
        default=OUTPUT_DIR,
        help=f"Path to save generated outputs, defaults to {OUTPUT_DIR}",
    )
    args = vars(parser.parse_args())

    main(**args)
