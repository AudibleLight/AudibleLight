#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, objects etc."""

import wave
from pathlib import Path
from typing import Union, List

import random
import numpy as np
import torch
from loguru import logger

MESH_UNITS = "meters"    # will convert to this if
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOREGROUND_DIR = "FSD50K_FMA_SMALL"
RIR_DIR = None
FORMAT = 'foa'    # First-order ambisonics by default
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "output"  # Directory to store the generated soundscapes
REF_DB = -65  # Reference decibel level for the background ambient noise. Try making this random too!
NSCAPES = 20
SEED = 42
SAMPLE_RATE = 44100    # Default to 44.1kHz sample rate


def write_wav(audio: np.ndarray, outpath: str, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Writes a mono audio array to a WAV file in 16-bit PCM format.

    The input must be a 1D float array. Values are expected in [-1, 1], and will
    be normalized if they exceed this range.
    """
    # Cast array to float64
    audio = np.asarray(audio, dtype=np.float64)
    # Sanity checking
    assert len(audio.shape) == 1, "Only mono audio supported"
    assert Path(outpath).parent.exists(), "Output directory must exist"
    # Check if normalization is needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        logger.warning(f"Audio file absolute max exceeds 1.0 ({round(max_val, 3)}), normalizing...")
        audio = audio / max_val
    # Catch cases where silent audio would lead to dividing by zero
    elif max_val == 0.:
        logger.warning("Audio is completely silent.")
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    # Coerce to 2D array
    audio_fmt = coerce2d(audio_int16).T
    # Write to WAV using wave module
    #  We use wave for dumping wav files, not scipy, as scipy depends on audioread which is no longer maintained
    #  See https://github.com/beetbox/audioread/issues/144
    with wave.open(outpath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_fmt.tobytes())


def coerce2d(array: Union[list[float], list[np.ndarray], np.ndarray]) -> np.ndarray:
    """Coerces an input type to a 2D array"""
    # Coerce list types to arrays
    if isinstance(array, list):
        array = np.array(array)
    # Convert 1D arrays to 2D
    if len(array.shape) == 1:
        array = np.array([array])
    if len(array.shape) != 2:
        raise ValueError(f"Expected a 1- or 2D array, but got {len(array.shape)}D array")
    return array


def seed_everything(seed: int = SEED) -> None:
    """Set the random seeds for libraries used by AudibleLight."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    # Possibly the root directory, but doesn't always work when running from the CLI for some reason
    poss_path = Path(__file__).parent.parent
    # The root directory should always have these files (this is pretty hacky)
    expected_files = ["audiblelight", "notebooks", "resources", "tests", "setup.py"]
    if all((poss_path / fp).exists() for fp in expected_files):
        return poss_path
    else:
        return Path.cwd()


def polar_to_cartesian(spherical_array: np.ndarray) -> np.ndarray:
    """Converts an array of spherical coordinates (azimuth°, polar°, radius) to Cartesian coordinates (XYZ)."""
    spherical_array = coerce2d(spherical_array)
    # Convert azimuth + elevation to radians
    azimuth_rad = np.deg2rad(spherical_array[:, 0])  # phi
    polar_rad = np.deg2rad(spherical_array[:, 1])    # theta, polar angle from z-axis
    # No need to do this for radius
    r = spherical_array[:, 2]
    # Express everything in cartesian form
    x = r * np.sin(polar_rad) * np.cos(azimuth_rad)
    y = r * np.sin(polar_rad) * np.sin(azimuth_rad)
    z = r * np.cos(polar_rad)
    # Stack into a 2D array of shape (n_capsules, 3)
    return np.column_stack((x, y, z))


def cartesian_to_polar(cartesian_array: np.ndarray) -> np.ndarray:
    """Converts an array of Cartesian coordinates (XYZ) to spherical coordinates (azimuth°, polar°, radius)."""
    cartesian_array = coerce2d(cartesian_array)
    # Unpack everything
    x = cartesian_array[:, 0]
    y = cartesian_array[:, 1]
    z = cartesian_array[:, 2]
    # Compute radius using the classic equation
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    assert r > 0, f"Expected radius > 0, but got radius = {r}"
    # Get azimuth and polar in radians first, then convert to degrees
    azimuth = np.rad2deg(np.arctan2(y, x))  # φ, angle in x-y plane from x-axis
    polar = np.rad2deg(np.arccos(z / r))  # θ, angle from z-axis
    # Ensure azimuth is in [0, 360)
    azimuth = np.mod(azimuth, 360)
    # Stack everything back into a 2D array of shape (n_capsules, 3)
    return np.column_stack((azimuth, polar, r))


def center_coordinates(cartesian_array: np.ndarray) -> np.ndarray:
    """Take a dictionary of cartesian coordinates, find the center, and subtract to center all coordinates around 0"""
    # Shape (3,)
    c_mean = np.mean(cartesian_array, axis=0)
    # Shape (n_capsules, 3)
    return cartesian_array - c_mean


def check_all_lens_equal(*iterables) -> bool:
    """
    Returns True if all iterables have the same length, False otherwise
    """
    return len({len(i) for i in iterables}) == 1


def list_all_directories(root_dir: str) -> List[str]:
    """
    Recursively return all directory paths under root_dir, including nested subdirectories.
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise ValueError(f"Directory '{root_dir}' does not exist")
    
    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' is not a directory")
    
    return [str(p.resolve()) for p in root_path.rglob('*') if p.is_dir()]
