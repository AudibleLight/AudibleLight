#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, objects etc."""

from pathlib import Path
from typing import Union

import random
import numpy as np
import torch

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


def foa_to_simple_stereo(audio: np.ndarray, angle: int = 0) -> np.ndarray:
    """Simple conversion of FOA ambisonic audio to stereo. We should replace this with something more complex later!"""
    assert audio.shape[0] == 4, "Input must be in FOA format (WXYZ)"
    w, x, y, _ = audio
    angle = angle * np.pi / 180
    left = w + 0.7071 * (x * np.cos(angle) + y * np.sin(angle))
    right = w + 0.7071 * (x * np.cos(-angle) + y * np.sin(-angle))
    return np.vstack((left, right))


def pad2d(iter2d: list[np.ndarray]) -> np.ndarray:
    """Pads a list of 2D arrays to the same length"""
    # Get the length of the longest array
    max_length = max([ir.shape[-1] for ir in iter2d])
    padded_iters = []
    # Iterate over all the arrays
    for ir in iter2d:
        # Pad short arrays
        if ir.shape[-1] < max_length:
            padded = np.pad(ir, ((0, 0), (0, max_length - ir.shape[-1])), mode='constant')
        # Truncate long ones
        else:
            padded = ir[:, :max_length]
        padded_iters.append(padded)
    return np.array(padded_iters)


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
