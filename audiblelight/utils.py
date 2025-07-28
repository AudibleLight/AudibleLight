#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, objects etc."""

import inspect
import json
import os
import random
import wave
from functools import wraps
from pathlib import Path
from typing import Union, Any, Protocol, Callable, List, Optional

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
MAX_PLACE_ATTEMPTS = 100    # Max number of times we'll attempt to place a source or microphone before giving up

NUMERIC_DTYPES = (int, float, complex, np.integer, np.floating)     # used for isinstance(x, ...) checking
Numeric = Union[int, float, complex, np.integer, np.floating]    # used as a typehint

AUDIO_EXTS = ("wav", "mp3", "mpeg4", "m4a", "flac", "aac")


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
    assert np.all(r > 0), f"Expected radius > 0, but got radius = {r}"
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


def sanitise_filepath(filepath: Any) -> Path:
    """
    Validate that a filepath exists on the disk and coerce to a `Path` object
    """
    if isinstance(filepath, (str, Path)):
        # Coerce string types to Path
        if isinstance(filepath, str):
            filepath = Path(filepath)
        # Raise a nicer error when the file can't be found
        if not filepath.is_file():
            raise FileNotFoundError(f"Cannot find file at {filepath}, does it exist?")
        else:
            return filepath
    else:
        raise TypeError(f"Expected filepath to be either a string or Path object, but got {type(filepath)}")


def sanitise_positive_number(x: Any) -> Union[float, None]:
    """
    Validate that an input is a positive numeric input and coerce to a `float`
    """
    if isinstance(x, NUMERIC_DTYPES) and not isinstance(x, bool):
        if x >= 0.:
            return float(x)
        else:
            raise ValueError(f"Expected a positive numeric input, but got {x}")
    else:
        raise TypeError("Expected a positive numeric input, but got {}".format(type(x)))


def sanitise_coordinates(x: Any) -> Union[np.ndarray, None]:
    """
    Validate that an input is an array of coordinates (i.e., [X, Y, Z]) with the expected shape
    """
    if isinstance(x, (np.ndarray, list)):
        if isinstance(x, list):
            x = np.asarray(x)
        if x.shape != (3,):
            raise ValueError(f"Expected a shape of (3,), but got {x.shape}")
        return x
    else:
        raise TypeError("Expected a list or array input, but got {}".format(type(x)))


class DistributionLike(Protocol):
    """
    Typing protocol for any distribution-like object.

    Must expose an `rvs()` method that returns a single random variate as a float (or float-compatible number).
    """

    def rvs(self, *args: Any, **kwargs: Any) -> Numeric:
        ...


class DistributionWrapper:
    """
    Wraps a callable (e.g. a function) as a distribution-like object with an `rvs()` method.
    """

    def __init__(self, distribution: Callable):
        self.distribution = distribution

    def rvs(self, *_: Any, **__: Any) -> Numeric:
        return self.distribution()

    def __call__(self) -> Numeric:
        """Makes the wrapper itself callable like the original."""
        return self.rvs()


def sanitise_distribution(x: Any) -> Union[DistributionLike, None]:
    """
    Validate that an input is a scipy distribution-like object, a callable returning floats, or None.
    """
    if x is None:
        return x

    # Otherwise, object is a scipy-like distribution
    elif hasattr(x, "rvs") and callable(x.rvs):
        return x

    # Otherwise, input is a function that might return random numbers
    elif callable(x):
        # Try and get a value from the function
        try:
            test_sample = x()
        except Exception as e:
            raise TypeError("Callable could not be evaluated during distribution validation") from e
        # If we get a numeric value back from the function, wrap it up so we have the same API as a scipy distribution
        if isinstance(test_sample, NUMERIC_DTYPES):
            return DistributionWrapper(x)
        else:
            raise TypeError("Callable must return a numeric value to be used as a distribution")

    # Otherwise, we cannot evaluate what the input is
    else:
        raise TypeError(f"Expected a distribution-like object or a callable returning floats, but got: {type(x)}")


def get_default_alias(prefix: str, objects: dict[str, Any], zfill_ints: int = 3) -> str:
    """
    Returns a default alias for a microphone, source, event...

    The alias is constructed in the form "{prefix}{idx}", where `prefix` is a required argument and `idx` is determined
    by the number of `objects` already present (e.g., the number of current microphones), left-padded with the number
    of `zfill_ints` (defaults to 3).

    Returns:
        str: the default alias

    Examples:
        >>> default_alias = get_default_alias("mic", {"mic000": "", "mic001": ""})
        >>> print(default_alias)
        mic002

    """
    n_current_objs = len(objects)
    test_alias = f"{prefix}{str(n_current_objs).zfill(zfill_ints)}"
    if test_alias in objects:
        raise KeyError(f"Alias {test_alias} already exists in dictionary!")
    return test_alias


def repr_as_json(cls: object) -> str:
    """
    Used for `__repr__` methods; dumps `self.to_dict()` to a nicely formatted JSON string.
    """
    if hasattr(cls, "to_dict") and callable(cls.to_dict):
        return json.dumps(cls.to_dict(), indent=4, ensure_ascii=False, sort_keys=False)
    else:
        raise AttributeError(f"Class {cls.__name__} has no attribute 'to_dict'")


def update_state(func: Callable):
    """
    Decorator function that will update a `WorldState` and all objects in it. Should be run after any
    method that changes the state, e.g. `add_microphone`, `add_emitter`.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._update()
        return result
    return wrapper


def list_all_directories(root_dir: Union[str, Path]) -> List[str]:
    """
    Recursively return all directory paths under root_dir, including nested subdirectories.
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory '{root_dir}' does not exist")

    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' is not a directory")

    return [str(p.resolve()) for p in root_path.rglob('*') if p.is_dir()]


def list_deepest_directories(root_dir: Union[str, Path]) -> List[str]:
    """
    Return only the deepest (leaf) directories under root_dir.
    A deepest directory is one that is not a parent of any other directory.
    """
    all_dirs = sorted([Path(p) for p in list_all_directories(root_dir)], key=lambda p: len(str(p)))
    deepest_dirs = []

    for d in all_dirs:
        # If no other dir in the set starts with this path + separator, it's a leaf
        if not any(other != d and str(other).startswith(str(d) + os.sep) for other in all_dirs):
            deepest_dirs.append(str(d.resolve()))

    return deepest_dirs


def list_innermost_directory_names(root_dir: Union[str, Path]) -> List[str]:
    """
    Return only the names of the innermost (leaf) directories under root_dir.

    Returns:
        List[str]: A list of directory names (not full paths) of the deepest directories.
    """
    deepest_paths = list_deepest_directories(root_dir)
    return [Path(path).name for path in deepest_paths]


def list_innermost_directory_names_unique(root_dir: Union[str, Path]) -> set:
    """
    Return only the unique names of the innermost (leaf) directories under root_dir.

    Returns:
        set: A set of unique directory names (not full paths) of the deepest directories.
    """
    deepest_paths = list_deepest_directories(root_dir)
    return {Path(path).name for path in deepest_paths}


def sample_distribution(
        distribution: Union[DistributionLike, Callable, None] = None,
        override: Union[Numeric, None] = None
) -> float:
    """
    Samples from a probability distribution or returns a provided override
    """
    # Callable functions are wrapped such that they have a `.rvs` method
    distribution = sanitise_distribution(distribution)
    if distribution is None and override is None:
        raise ValueError("Must provide either a probability distribution to sample from or an override")
    elif override is None:
        return distribution.rvs()
    else:
        if isinstance(override, NUMERIC_DTYPES):
            return override
        else:
            raise TypeError(f"Expected a numeric input for `override` but got {type(override)}")


def validate_kwargs(func: Callable, **kwargs) -> None:
    """
    Validates that the given kwargs are acceptable keyword arguments for the provided function.

    Raises:
        TypeError: if `func` is not callable.
        ValueError: if `func` has no keyword arguments.
        AttributeError: if a kwarg in `kwargs` is an invalid kwarg for `func.
    """
    if not callable(func):
        raise TypeError("`func` must be a callable")

    sig = inspect.signature(func)
    params = sig.parameters

    # If function accepts arbitrary kwargs, no need to validate
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return

    valid_kwargs = {
        name for name, param in params.items()
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    }

    if not valid_kwargs:
        raise ValueError("`func` must have at least one named keyword argument")

    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise AttributeError(f"`{kwarg}` is not a valid keyword argument for `{func.__name__}`")


def validate_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    """
    Compares the shapes of two arrays and validate that they are compatible.

    Shapes should be a tuple of integers (i.e., returned with `np.array([]).shape`. They can be any length and are
    implicitly padded with `None` in cases where they have a different number of dimensions.

    Raises:
        ValueError: if any corresponding non-None dimensions differ.
    """
    # Pad the shapes so they are the same length
    max_len = max(len(shape_a), len(shape_b))
    padded_a = shape_a + (None,) * (max_len - len(shape_a))
    padded_b = shape_b + (None,) * (max_len - len(shape_b))

    for i, (a, b) in enumerate(zip(padded_a, padded_b)):
        # Implicitly skip over `None` values
        if a is not None and b is not None and a != b:
            raise ValueError(f"Incompatible shapes at index {i}: {a} != {b} (full shapes: {padded_a} vs {padded_b})")


def db_to_multiplier(db: Numeric, x: Numeric) -> float:
    """
    Calculates the multiplier factor from a decibel (dB) value that, when applied to x, adjusts its amplitude to
    reflect the specified dB. The relationship is based on the formula 20 * log10(factor * x) ≈ db.

    Arguments:
        db (float): The target decibel change to be applied.
        x  (float): The original amplitude of x

    Returns:
        float: The multiplier factor.
    """
    return 10 ** (db / 20) / x
