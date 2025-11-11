#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, objects etc."""

import inspect
import json
import os
import random
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from time import time
from typing import Any, Callable, Generator, Optional, Union

import numpy as np
import torch
from loguru import logger

from audiblelight.custom_types import (
    NUMERIC_DTYPES,
    DistributionLike,
    DistributionWrapper,
    Numeric,
)

# Units for mesh
# Device for any torch code: default to GPU, then MPS (on macOS), then CPU
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
# Seed used for randomisation
SEED = 42
# Useful as a constant for tolerance checking, when `utils.tiny(...)` is going to be too small
SMALL = 1e-4


@contextmanager
def timer(name: str) -> Generator[None, Any, None]:  # pragma: no cover
    """
    Log how long it takes to execute the provided block.

    Examples:
        Define a function with timing
        >>> @timer("add")
        >>> def time_me(a, b):
        >>>     return a + b
        >>>
        >>> time_me(1, 2)
    """
    start = time()
    try:
        yield
    except Exception as e:
        end = time()
        logger.warning(f"Took {end - start:.2f} seconds to {name} and raised {e}.")
        raise e
    else:
        end = time()
        logger.debug(f"Took {end - start:.2f} seconds to {name}.")


def coerce2d(array: Union[list[float], list[np.ndarray], np.ndarray]) -> np.ndarray:
    """Coerces an input type to a 2D array"""
    # Coerce list types to arrays
    if isinstance(array, list):
        array = np.array(array)
    # Convert 1D arrays to 2D
    if len(array.shape) == 1:
        array = np.array([array])
    if len(array.shape) != 2:
        raise ValueError(
            f"Expected a 1- or 2D array, but got {len(array.shape)}D array"
        )
    return array


def seed_everything(seed: int = SEED) -> None:  # pragma: no cover
    """Set the random seeds for libraries used by AudibleLight."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)


# noinspection PyUnresolvedReferences
def get_project_root() -> Path:  # pragma: no cover
    """Returns the root directory of the project."""
    return resources.files("audiblelight").parent


def polar_to_cartesian(spherical_array: np.ndarray) -> np.ndarray:
    """
    Converts an array of spherical coordinates (azimuth°, polar°, radius) to Cartesian coordinates (XYZ).

    Assumptions:
        Azimuth: [-180, 180), increasing counter-clockwise from front (i.e, azimuth=90 == left)
        Elevation: [-90, 90], 0 = horizontal, 90 = up, -90 = down.
        Radius: unbounded, in metres.
    """
    spherical_array = coerce2d(spherical_array)

    # Sanity check inputs
    assert np.all(np.abs(spherical_array[:, 0] <= 180)), "Invalid elevation angle"
    assert np.all(np.abs(spherical_array[:, 1] <= 90)), "Invalid elevation angle"

    # Convert azimuth + elevation to radians
    azimuth_rad = np.deg2rad(spherical_array[:, 0])  # phi
    elevation_rad = np.deg2rad(spherical_array[:, 1])  # theta, polar angle from z-axis

    # No need to do this for radius
    r = spherical_array[:, 2]

    # Express everything in cartesian form
    x = r * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = r * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = r * np.sin(elevation_rad)

    # Stack into a 2D array of shape (n_capsules, 3)
    return np.column_stack((x, y, z))


def cartesian_to_polar(cartesian_array: np.ndarray) -> np.ndarray:
    """
    Converts an array of Cartesian coordinates (XYZ) to spherical coordinates (azimuth°, polar°, radius).

    Assumptions:
        Azimuth: [-180, 180), increasing counter-clockwise from front (i.e, azimuth=90 == left)
        Elevation: [-90, 90], 0 = horizontal, 90 = up, -90 = down.
        Radius: unbounded, in metres.
    """
    cartesian_array = coerce2d(cartesian_array)

    # Unpack everything
    x = cartesian_array[:, 0]
    y = cartesian_array[:, 1]
    z = cartesian_array[:, 2]

    # Compute radius using the classic equation
    r = np.sqrt(x**2 + y**2 + z**2)
    assert np.all(r > 0), f"Expected radius > 0, but got radius = {r}"

    # Get azimuth and polar in radians first, then convert to degrees
    azimuth = np.rad2deg(np.arctan2(y, x))  # φ, angle in x-y plane from x-axis
    elevation = np.rad2deg(np.arcsin(z / r))  # θ, angle from z-axis in [-90, 90]

    # Stack everything back into a 2D array of shape (n_capsules, 3)
    return np.column_stack((azimuth, elevation, r))


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
            raise FileNotFoundError(
                f"Cannot find file at {str(filepath)}, does it exist?"
            )
        else:
            return filepath
    else:
        raise TypeError(
            f"Expected filepath to be either a string or Path object, but got {type(filepath)}"
        )


def sanitise_filepaths(filepaths: list[Any]) -> list[Path]:
    """
    Equivalent to [sanitise_filepath(fp) for fp in filepaths]
    """
    return [sanitise_filepath(fp) for fp in filepaths]


def sanitise_directory(directory: Any, create_if_missing: bool = False) -> Path:
    """
    Validate that a directory exists on the disk and coerce to a `Path` object.

    If `create_if_missing` and the folder does not exist, it will be created
    """
    if isinstance(directory, (str, Path)):
        # Coerce string types to Path
        if isinstance(directory, str):
            directory = Path(directory)
        # Raise a nicer error when the file can't be found
        if not directory.is_dir():
            if create_if_missing:
                directory.mkdir(parents=True, exist_ok=True)
                return directory
            else:
                raise FileNotFoundError(
                    f"Cannot find directory at {str(directory)}, does it exist?"
                )
        else:
            if not any(directory.iterdir()):
                logger.warning(
                    f"Directory {str(directory)} does not contain any files!"
                )
            return directory
    else:
        raise TypeError(
            f"Expected directory to be either a string or Path object, but got {type(directory)}"
        )


def sanitise_directories(
    directories: list[Any], create_if_missing: bool = False
) -> list[Path]:
    """
    Equivalent to [sanitise_directory(dir) for dir in directories]
    """
    return [sanitise_directory(dir_, create_if_missing) for dir_ in directories]


def sanitise_positive_number(x: Any, cast_to: type = float) -> Optional[Numeric]:
    """
    Validate that an input is a positive numeric input and coerce to `cast_to` (default: float)
    """
    if isinstance(x, NUMERIC_DTYPES) and not isinstance(x, bool):
        if x >= 0.0:
            return cast_to(x)
        else:
            raise ValueError(f"Expected a positive numeric input, but got {x}")
    else:
        raise TypeError("Expected a positive numeric input, but got {}".format(type(x)))


def sanitise_coordinates(x: Any) -> Optional[np.ndarray]:
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


def sanitise_distribution(
    x: Any,
) -> Optional[Union[DistributionLike, DistributionWrapper]]:
    """
    Validate that an input is a scipy distribution-like object, a callable returning floats, or None.
    """
    if x is None:
        return x

    # Otherwise, object is a scipy-like distribution
    elif hasattr(x, "rvs") and callable(x.rvs):
        return DistributionWrapper(x.rvs)

    # Otherwise, input is a function that might return random numbers
    elif callable(x):
        # Try and get a value from the function
        try:
            test_sample = x()
        except Exception as e:
            raise TypeError(
                "Callable could not be evaluated during distribution validation"
            ) from e
        # If we get a numeric value back from the function, wrap it up so we have the same API as a scipy distribution
        if isinstance(test_sample, NUMERIC_DTYPES):
            return DistributionWrapper(x)
        else:
            raise TypeError(
                "Callable must return a numeric value to be used as a distribution"
            )

    # Otherwise, we cannot evaluate what the input is
    else:
        raise TypeError(
            f"Expected a distribution-like object or a callable returning floats, but got: {type(x)}"
        )


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


def list_all_directories(root_dir: Union[str, Path]) -> list[str]:
    """
    Recursively return all directory paths under root_dir, including nested subdirectories.
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory '{root_dir}' does not exist")

    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' is not a directory")

    return [str(p.resolve()) for p in root_path.rglob("*") if p.is_dir()]


def list_deepest_directories(root_dir: Union[str, Path]) -> list[str]:
    """
    Return only the deepest (leaf) directories under root_dir.
    A deepest directory is one that is not a parent of any other directory.
    """
    all_dirs = sorted(
        [Path(p) for p in list_all_directories(root_dir)], key=lambda p: len(str(p))
    )
    deepest_dirs = []

    for d in all_dirs:
        # If no other dir in the set starts with this path + separator, it's a leaf
        if not any(
            other != d and str(other).startswith(str(d) + os.sep) for other in all_dirs
        ):
            deepest_dirs.append(str(d.resolve()))

    return deepest_dirs


def list_innermost_directory_names(root_dir: Union[str, Path]) -> list[str]:
    """
    Return only the names of the innermost (leaf) directories under root_dir.

    Returns:
        list[str]: A list of directory names (not full paths) of the deepest directories.
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


# noinspection PyUnreachableCode
def sample_distribution(
    distribution: Union[DistributionLike, Callable, None] = None,
    override: Union[Numeric, None] = None,
) -> float:
    """
    Samples from a probability distribution or returns a provided override
    """
    # Callable functions are wrapped such that they have a `.rvs` method
    distribution = sanitise_distribution(distribution)
    if distribution is None and override is None:
        raise ValueError(
            "Must provide either a probability distribution to sample from or an override"
        )
    elif override is None:
        return distribution.rvs()
    else:
        if isinstance(override, NUMERIC_DTYPES):
            return override
        else:
            raise TypeError(
                f"Expected a numeric input for `override` but got {type(override)}"
            )


def get_valid_kwargs(func: Callable) -> set[str]:
    """
    Gets the names of all valid keyword arguments for the provided function.

    Note that this function assumes that `func` takes in an arbitrary number of keyword arguments. It is not designed
    to be used in cases where (for instance) `func` accepts only positional arguments or `*args`.

    Arguments:
        func: a function to call

    Raises:
        TypeError: if `func` is not callable.
        ValueError: if `func` has no keyword arguments.
        AttributeError: if a kwarg in `kwargs` is an invalid kwarg for `func`.

    Returns:
        set[str]: the names of all valid keyword arguments for the provided function.
    """

    if not callable(func):
        raise TypeError("`func` must be a callable")

    sig = inspect.signature(func)
    params = sig.parameters

    # If function accepts arbitrary kwargs, return empty set
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return {}

    valid_kwargs = {
        name
        for name, param in params.items()
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    }

    return valid_kwargs


# noinspection PyUnreachableCode
def validate_kwargs(func: Callable, **kwargs) -> None:
    """
    Validates that the given kwargs are acceptable keyword arguments for the provided function.

    Note that this function assumes that `func` takes in an arbitrary number of keyword arguments. It is not designed
    to be used in cases where (for instance) `func` accepts only positional arguments or `*args`.

    Arguments:
        func: a function to call
        kwargs: keyword arguments to validate for `func`

    Raises:
        TypeError: if `func` is not callable.
        ValueError: if `func` has no keyword arguments.
        AttributeError: if a kwarg in `kwargs` is an invalid kwarg for `func`.
    """
    valid_kwargs = get_valid_kwargs(func)

    if not valid_kwargs:
        raise ValueError("`func` must have at least one named keyword argument")

    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise AttributeError(
                f"`{kwarg}` is not a valid keyword argument for `{func.__name__}`"
            )


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
            raise ValueError(
                f"Incompatible shapes at index {i}: {a} != {b} (full shapes: {padded_a} vs {padded_b})"
            )


def generate_linear_trajectory(
    xyz_start: np.ndarray, xyz_end: np.ndarray, n_points: int
) -> np.ndarray:
    """
    Generate a linear trajectory between a start and end coordinate given a particular number of points
    """
    return np.array(
        [list(np.linspace(xyz_start, xyz_end, n_points)[i]) for i in range(n_points)]
    )


def generate_semicircular_trajectory(
    xyz_start: np.ndarray, xyz_end: np.ndarray, n_points: int
) -> np.ndarray:
    """
    Generate a semicircular trajectory (arc) between a start and end coordinate given a particular number of points
    """
    # Vector from start to end
    start_to_end_vec = xyz_end - xyz_start
    midpoint = xyz_start + start_to_end_vec / 2

    # Radius of the circle
    radius = np.linalg.norm(start_to_end_vec) / 2

    # Normal vector to the plane containing the circle
    matches: np.ndarray = start_to_end_vec == [0, 0, 0]
    if matches.all():
        normal_vector = np.array([1, 0, 0])  # default normal vector
    else:
        normal_vector = np.array([1, 0, 0])  # initial guess
        if np.cross(normal_vector, start_to_end_vec).any():
            normal_vector = np.cross(start_to_end_vec, normal_vector)
        else:
            normal_vector = np.cross(start_to_end_vec, [0, 1, 0])
    normal_vector /= np.linalg.norm(normal_vector)

    # Finding two orthogonal unit vectors in the plane of the circle
    vec1 = start_to_end_vec / (2 * radius)
    vec2 = np.cross(normal_vector, vec1)

    # Generate points on the semicircle
    angle_range = np.linspace(np.pi, 0, n_points)  # Reversed angle range
    circle_points = []
    for angle in angle_range:
        point = midpoint + radius * (np.cos(angle) * vec1 + np.sin(angle) * vec2)
        circle_points.append(list(point))

    return np.array(circle_points)


def generate_random_trajectory(
    xyz_start: np.ndarray,
    max_step: Numeric,
    n_points: int,
) -> np.ndarray:
    """
    Generate a 3D random walk from `xyz_start` with `n_points` steps, such that no step exceeds `max_step`.
    """
    if max_step <= 0.0:
        raise ValueError(f"Maximum step must be greater than 0 but got {max_step}")

    # Generate random directions in 3D and normalize to unit vectors
    directions = np.random.normal(size=(n_points - 1, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_vectors = directions / norms

    # Generate random step lengths in [0, step_limit]
    step_lengths = np.random.uniform(0, max_step, size=(n_points - 1, 1))

    # Scale unit vectors by step lengths
    steps = unit_vectors * step_lengths

    # Compute random walk as the cumulative sum of every step, added to the starting position
    trajectory = xyz_start + np.cumsum(steps, axis=0)

    # Stack to add the starting position
    return np.vstack([xyz_start, trajectory])


def generate_sinusoidal_trajectory(
    xyz_start: np.ndarray,
    xyz_end: np.ndarray,
    n_points: int,
    amplitude: float = None,
    frequency: int = None,
) -> np.ndarray:
    """
    Generate a sinusoidal trajectory between a start and end coordinate given a particular number of points.
    Amplitude and frequency of the sine wave are randomly sampled if not provided.
    """
    # Amplitude: between 10 cm and 500 cm
    if amplitude is None:
        amplitude = np.random.uniform(0.01, 0.5)
    # Frequency (number of complete sine waves): between 1 and 3
    if frequency is None:
        frequency = np.random.randint(1, 4)

    baseline = xyz_end - xyz_start
    length = np.linalg.norm(baseline)

    direction = baseline / length

    if np.allclose(direction, [0, 0, 1]):
        perp1 = np.array([1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 0, 1])
        perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    t = np.linspace(0, 1, n_points)
    points = xyz_start + np.outer(t, baseline)

    sine_wave = np.sin(2 * np.pi * frequency * t)
    offset = amplitude * (np.outer(sine_wave, perp1) + np.outer(sine_wave, perp2))
    points += offset

    return points


def generate_sawtooth_trajectory(
    xyz_start: np.ndarray,
    xyz_end: np.ndarray,
    n_points: int,
    amplitude: float = None,
    frequency: int = None,
    plane: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a sawtooth (zigzag) trajectory between `start` and `end` points.
    Amplitude, frequency, and plane are randomly sampled if not provided.
    """
    # Amplitude: between 10 cm and 500 cm
    if amplitude is None:
        amplitude = np.random.uniform(0.01, 0.5)
    # Frequency (number of complete zigzags): between 1 and 3
    if frequency is None:
        frequency = np.random.randint(1, 4)

    if plane is None:
        plane = np.random.choice(["xy", "xz", "yz"])

    # Linearly interpolate the straight path
    t = np.linspace(0, 1, n_points)
    trajectory = (1 - t)[:, None] * xyz_start + t[:, None] * xyz_end

    # Create zigzag wave (sharp turns)
    zigzag_wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

    # Apply zigzag to one of the axes depending on plane
    if plane == "xy":
        trajectory[:, 0] += zigzag_wave  # Zig along X
    elif plane == "xz":
        trajectory[:, 0] += zigzag_wave  # Zig along X
    elif plane == "yz":
        trajectory[:, 1] += zigzag_wave  # Zig along Y
    else:
        raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'yz'.")

    return trajectory


def pad_or_truncate_audio(
    audio: np.ndarray, desired_samples: Numeric, pad_mode: str = "constant"
) -> np.ndarray:
    """
    Pads or truncates audio with desired number of samples.

    Arguments:
        audio (np.ndarray): array to pad or truncate
        desired_samples (Numeric): desired number of samples for the target audio
        pad_mode (str): mode to use with `np.pad`, defaults to constant
    """
    # Audio is too short, needs padding
    if audio.shape[1] < desired_samples:
        return np.pad(
            audio, ((0, 0), (0, desired_samples - audio.shape[1])), mode=pad_mode
        )
    # Audio is too long, needs truncating
    elif audio.shape[1] > desired_samples:
        return audio[:, :desired_samples]
    # Audio is just right
    else:
        return audio


def tiny(x: Union[float, np.ndarray]) -> Numeric:
    """
    Compute the tiny-value corresponding to an input's data type, preventing underflow and divide-by-zero errors
    """
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny


def coerce_nested_inputs(inp: Any) -> Any:
    """
    Coerce nested dtypes for JSON serialisation
    """
    if isinstance(inp, dict):
        return {k: coerce_nested_inputs(v) for k, v in inp.items()} if inp else None
    elif isinstance(inp, np.ndarray):
        return inp.tolist()
    else:
        return inp
