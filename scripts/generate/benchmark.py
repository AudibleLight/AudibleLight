#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmark `AudibleLight` by generating N scenes with X static and Y moving sound events."""


import argparse
import os
import random
from pathlib import Path
from time import time

from generate_with_random_events import main as make_a_scene
from loguru import logger
from tqdm import tqdm

from audiblelight import config, utils

FG_DIR = utils.get_project_root() / "resources/soundevents"
BG_DIR = FG_DIR / "domesticSounds"
MESH_DIR = utils.get_project_root() / "resources/meshes"
MESHES = list(MESH_DIR.rglob("*.glb"))

OUTPUT_DIR = utils.get_project_root() / "spatial_scenes"

# Distributions to sample
STATIC_EVENTS = utils.sanitise_distribution(
    lambda: random.choice(range(config.MIN_STATIC_EVENTS, config.MAX_STATIC_EVENTS))
)
MOVING_EVENTS = utils.sanitise_distribution(
    lambda: random.choice(range(config.MIN_MOVING_EVENTS, config.MIN_MOVING_EVENTS))
)


def main(n_scenes: int, outdir: str):
    # Create the output folder if it doesn't currently exist
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outdir = Path(outdir)

    # Start iterating to create the required number of scenes
    logger.info(f"Generating {n_scenes} scenes...")
    full_start = time()
    for scene_idx in tqdm(range(n_scenes), desc="Running benchmark..."):

        # Choose a random mesh
        mesh = random.choice(MESHES)

        # Output folder is just the index to prevent overwriting when we use the same mesh multiple times
        #  A folder with this name will be created inside `make_a_scene`, no need to do this now
        output_dir = outdir / f"scene_{str(scene_idx).zfill(3)}"

        # Skip over existing files
        if os.path.isdir(output_dir):
            continue

        # Make the scene with the mesh
        make_a_scene(
            duration=config.SCENE_DURATION,
            n_static=STATIC_EVENTS.rvs(),
            n_moving=MOVING_EVENTS.rvs(),
            max_overlap=config.MAX_OVERLAP,
            micarray=config.MIC_ARRAY_TYPE,
            output_folder=output_dir,
            fg_folder=FG_DIR,
            mesh_path=mesh,
            ref_db=config.DEFAULT_REF_DB,
            min_snr=config.MIN_EVENT_SNR,
            max_snr=config.MAX_EVENT_SNR,
            min_velocity=config.MIN_EVENT_VELOCITY,
            max_velocity=config.MAX_EVENT_VELOCITY,
            min_resolution=config.MIN_EVENT_RESOLUTION,
            max_resolution=config.MAX_EVENT_RESOLUTION,
            min_duration=config.MIN_EVENT_DURATION,
            max_duration=config.MAX_EVENT_DURATION,
            bg_folder=BG_DIR,
        )

    # Log the time taken
    full_end = time() - full_start
    logger.info(f"Finished in {full_end:.4f} seconds.")
    logger.info(f"Average time per scene: {(full_end / n_scenes):.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarking by generating multiple scenes with different meshes and events."
    )
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=config.N_SCENES,
        help=f"Number of scenes to generate, defaults to {config.N_SCENES}",
    )
    parser.add_argument(
        "--outdir",
        type=int,
        default=OUTPUT_DIR,
        help=f"Path to save generated outputs, defaults to {OUTPUT_DIR}",
    )
    args = vars(parser.parse_args())

    main(**args)
