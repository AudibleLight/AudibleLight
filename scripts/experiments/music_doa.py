#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests DOA for simulated sound events with MUSIC"""

import argparse
import random
from pathlib import Path

import numpy as np
from loguru import logger
from pyroomacoustics.doa import MUSIC
from scipy import stats
from scipy.signal import stft
from tqdm import tqdm

from audiblelight import config, utils
from audiblelight.core import Scene

# Filepaths, directories, etc.
FG_DIR = utils.get_project_root() / "resources/soundevents"
MESH_DIR = utils.get_project_root() / "resources/meshes"
MESHES = list(MESH_DIR.rglob("*.glb"))

# Types of noise we'll add
NOISE_TYPES = ["pink", "brown", "red", "blue", "white", "violet"]

MIC_TYPE = "eigenmike32"


def create_scene(mesh_path: Path) -> Scene:
    return Scene(
        duration=config.SCENE_DURATION,
        mesh_path=Path(mesh_path),
        scene_start_dist=stats.uniform(0.0, config.SCENE_DURATION - 1),
        event_start_dist=None,
        event_duration_dist=stats.uniform(
            config.MIN_EVENT_DURATION,
            config.MAX_EVENT_DURATION - config.MIN_EVENT_DURATION,
        ),
        snr_dist=stats.uniform(
            config.MIN_EVENT_SNR, config.MAX_EVENT_SNR - config.MIN_EVENT_SNR
        ),
        fg_path=Path(FG_DIR),
        max_overlap=1,
        ref_db=config.REF_DB,
        state_kwargs=dict(
            add_to_context=False, rlr_kwargs=dict(sample_rate=config.SAMPLE_RATE)
        ),
        allow_duplicate_audios=False,
    )


def apply_music(scene: Scene) -> MUSIC:
    # Coordinates of our capsules for the eigenmike
    l_: np.ndarray = scene.get_microphone("mic000").coordinates_absolute
    l_ = l_.T

    # Get the parameters
    fs = int(scene.sample_rate)
    num_sources = len(scene.get_events())
    freq_range = [300, 3500]

    # Create the MUSIC object
    #  Ensure azimuth is in range [-180, 180], increasing counter-clockwise
    #  Ensure colatitude is in range [-90, 90], where 0 == straight ahead
    music = MUSIC(
        L=l_,
        fs=fs,
        nfft=config.FFT_SIZE,
        # TODO: needs to be [-180, 180] once PR is merged
        azimuth=np.deg2rad(np.arange(360)),
        colatitude=np.deg2rad(np.arange(-90, 90)),
        num_sources=num_sources,
        dim=3,
    )

    # Compute the STFT
    stft_signals = stft(
        scene.audio["mic000"], fs=fs, nperseg=config.FFT_SIZE, noverlap=0, boundary=None
    )[2]

    # Locate the sources
    music.locate_sources(stft_signals, num_src=num_sources, freq_range=freq_range)

    return music


def main(n_scenes: int, microphone_type: str):
    angular_errors = []

    for _ in tqdm(range(n_scenes), desc="Generating scenes and applying MUSIC..."):
        # Choose a random mesh and create a scene
        mesh_path = random.choice(MESHES)
        scene = create_scene(mesh_path)

        # Add microphone type to the scene
        scene.add_microphone(microphone_type=microphone_type)

        # Add a single static event + background noise
        #  Event SNR will be sampled randomly from distribution
        scene.add_event(event_type="static")
        scene.add_ambience(noise=random.choice(NOISE_TYPES))

        # Run the simulation
        scene.generate(audio=False, metadata_json=False, metadata_dcase=False)

        # Apply MUSIC
        music = apply_music(scene)

        # Skip over cases where no azimuth/colatitude estimated
        if music.azimuth_recon is None or music.colatitude_recon is None:
            angular_errors.append(np.nan)

        # Compute estimated/actual azimuth/colatitude
        est_az = np.rad2deg(music.azimuth_recon)
        # est_col = np.rad2deg(music.colatitude_recon)
        act_az, act_col, _ = (
            scene.get_event(0).emitters[0].coordinates_relative_polar["mic000"][0]
        )

        angular_errors.append(abs(est_az - act_az))

    # Compute mean/SD angular error and log
    mean_angular_error = np.nanmean(angular_errors)
    std_angular_error = np.nanstd(angular_errors)

    logger.info(f"Mean angular error: {mean_angular_error:.3f}")
    logger.info(f"SD angular error: {std_angular_error:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests simulated DOA using MUSIC")
    parser.add_argument(
        "--n_scenes",
        type=int,
        help=f"Number of scenes to create, defaults to {config.N_SCENES}",
        default=1,
    )
    parser.add_argument(
        "--microphone_type",
        type=str,
        help=f"Microphone type to use, defaults to {MIC_TYPE}",
        default=MIC_TYPE,
    )
    args = vars(parser.parse_args())

    main(**args)
