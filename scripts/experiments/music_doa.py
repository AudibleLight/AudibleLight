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
DURATION = 10


def create_scene(mesh_path: Path) -> Scene:
    return Scene(
        duration=DURATION,
        sample_rate=config.SAMPLE_RATE,
        backend="rlr",
        backend_kwargs=dict(
            mesh=mesh_path,
            add_to_context=False,
        ),
        scene_start_dist=stats.uniform(0.0, DURATION - 1),
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
        ref_db=config.DEFAULT_REF_DB,
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
        azimuth=np.deg2rad(np.arange(-180, 180)),
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


def angular_error(
    pred_el: float, pred_az: float, true_el: float, true_az: float
) -> float:
    """
    Given predicted/ground-truth elevation + azimuth in radians, compute angular error in degrees
    """
    # Convert to unit vectors
    pred_vec = np.array(
        [
            np.cos(pred_el) * np.cos(pred_az),
            np.cos(pred_el) * np.sin(pred_az),
            np.sin(pred_el),
        ]
    )
    true_vec = np.array(
        [
            np.cos(true_el) * np.cos(true_az),
            np.cos(true_el) * np.sin(true_az),
            np.sin(true_el),
        ]
    )

    # Compute angular error: dot product of unit vectors, clipped to [-1, 1]
    dot_product = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg


def main(n_scenes: int, microphone_type: str):
    angular_errors = []

    for _ in tqdm(range(n_scenes), desc="Generating scenes and applying MUSIC..."):
        # Choose a random mesh and create a scene
        mesh_path = random.choice(MESHES)
        scene = create_scene(mesh_path)

        # Add microphone type to the scene
        scene.add_microphone(microphone_type=microphone_type)

        # Add a single static event
        #  Event SNR will be sampled randomly from distribution
        scene.add_event(event_type="static")

        # Run the simulation
        scene.generate(audio=False, metadata_json=False, metadata_dcase=False)

        # Apply MUSIC
        music = apply_music(scene)

        # Skip over cases where no azimuth/colatitude estimated
        if music.azimuth_recon is None or music.colatitude_recon is None:
            angular_errors.append(np.nan)

        # Compute actual azimuth/colatitude in radians
        act_az, act_col, _ = (
            scene.get_event(0).emitters[0].coordinates_relative_polar["mic000"][0]
        )
        act_az, act_col = np.deg2rad(act_az), np.deg2rad(act_col)

        # Compute angular error, in degrees
        error = angular_error(
            (np.pi / 2) - music.colatitude_recon[0],
            music.azimuth_recon[0],
            act_col,
            act_az,
        )
        angular_errors.append(error)

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
        default=config.N_SCENES,
    )
    parser.add_argument(
        "--microphone_type",
        type=str,
        help=f"Microphone type to use, defaults to {MIC_TYPE}",
        default=MIC_TYPE,
    )
    args = vars(parser.parse_args())

    main(**args)
