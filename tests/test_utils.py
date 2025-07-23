#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/utils.py"""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from scipy import stats

from audiblelight import utils


EXISTING_FILE = utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb"


@pytest.mark.parametrize(
    "spherical, expected",
    [
        # azimuth=0°, polar=0°, r=1 -> (0, 0, 1)
        (np.array([[0, 0, 1]]), np.array([[0, 0, 1]])),
        # azimuth=0°, polar=90°, r=1 -> (1, 0, 0)
        (np.array([[0, 90, 1]]), np.array([[1, 0, 0]])),
        # azimuth=90°, polar=90°, r=1 -> (0, 1, 0)
        (np.array([[90, 90, 1]]), np.array([[0, 1, 0]])),
        # azimuth=180°, polar=90°, r=1 -> (-1, 0, 0)
        (np.array([[180, 90, 1]]), np.array([[-1, 0, 0]])),
        # azimuth=0°, polar=180°, r=1 -> (0, 0, -1)
        (np.array([[0, 180, 1]]), np.array([[0, 0, -1]])),
        # Multiple points
        (
            np.array([
                [0, 0, 1],     # +Z
                [0, 90, 1],    # +X
                [90, 90, 1],   # +Y
                [180, 90, 1],  # -X
                [0, 180, 1]    # -Z
            ]),
            np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, -1]
            ])
        ),
    ]
)
def test_polar_to_cartesian(spherical, expected):
    result = utils.polar_to_cartesian(spherical)
    assert np.allclose(result, expected, atol=1e-4)
    # Go the other way
    assert np.allclose(utils.cartesian_to_polar(result), spherical, atol=1e-4)


@pytest.mark.parametrize(
    "cartesian, expected",
    [
        # (x, y, z) -> (azimuth°, polar°, r)
        (np.array([[0, 0, 1]]), np.array([[0, 0, 1]])),       # +Z axis
        (np.array([[1, 0, 0]]), np.array([[0, 90, 1]])),      # +X axis
        (np.array([[0, 1, 0]]), np.array([[90, 90, 1]])),     # +Y axis
        (np.array([[-1, 0, 0]]), np.array([[180, 90, 1]])),   # -X axis
        (np.array([[0, 0, -1]]), np.array([[0, 180, 1]])),    # -Z axis
        (
            np.array([
                [0, 0, 1],     # +Z
                [1, 0, 0],     # +X
                [0, 1, 0],     # +Y
                [-1, 0, 0],    # -X
                [0, 0, -1],    # -Z
            ]),
            np.array([
                [0, 0, 1],
                [0, 90, 1],
                [90, 90, 1],
                [180, 90, 1],
                [0, 180, 1],
            ])
        ),
    ]
)
def test_cartesian_to_polar(cartesian, expected):
    result = utils.cartesian_to_polar(cartesian)
    np.testing.assert_allclose(result, expected, atol=1e-4)
    # Go the other way
    assert np.allclose(utils.polar_to_cartesian(result), cartesian, atol=1e-4)


@pytest.mark.parametrize(
    "fpath,expected",
    [
        (EXISTING_FILE, Path(EXISTING_FILE)),
        (Path(EXISTING_FILE), Path(EXISTING_FILE)),
        ("a/broken/filepath", FileNotFoundError),
        (123456, TypeError)
    ]
)
def test_sanitise_filepath(fpath, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = utils.sanitise_filepath(fpath)
    else:
        assert expected == utils.sanitise_filepath(fpath)


@pytest.mark.parametrize(
    "num,expected",
    [
        (1, 1.),
        (0.5, 0.5),
        (-1/3, ValueError),
        ("asdf", TypeError)
    ]
)
def test_sanitise_positive_number(num, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = utils.sanitise_positive_number(num)
    else:
        assert expected == utils.sanitise_positive_number(num)


@pytest.mark.parametrize(
    "coords,expected",
    [
        ([0.5, 0.5, 0.5], np.array([0.5, 0.5, 0.5])),
        (np.array([0.1, 0.1]), ValueError),
        ("asdf", TypeError)
    ]
)
def test_sanitise_coordinates(coords, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = utils.sanitise_coordinates(coords)
    else:
        assert np.array_equal(expected, utils.sanitise_coordinates(coords))


@pytest.mark.parametrize(
    "dist,error",
    [
        # Scipy distributions should work ok out-of-the-box
        (stats.norm(0.5, 1.5), None),
        (stats.truncnorm(0.5, 1.5, 0.2, 1.7), None),
        (stats.uniform(0., 1.), None),
        # Callable that returns a random value -- ok
        (lambda: np.random.uniform(0., 1.), None),
        # Callable that will error out
        (lambda: 1 / 0, TypeError),
        # Callable that returns a string, or just a string -- nope
        (lambda: "asdf", TypeError),
        ("asdf", TypeError)
    ]
)
def test_sanitise_distribution(dist, error):
    if not error:
        # Sanitised distributions should have a `rvs` method which, when called, returns a number
        #  This is identical to how SciPy handles distributions
        #  We wrap non-SciPy objects in a `DistributionWrapper` class that adds the `rvs` method
        sanitised = utils.sanitise_distribution(dist)
        assert hasattr(sanitised, "rvs")
        assert callable(sanitised.rvs)
        assert isinstance(sanitised.rvs(), (int, float, complex))
    else:
        with pytest.raises(error):
            _ = utils.sanitise_distribution(dist)


@pytest.mark.parametrize(
    "prefix,objects,expected",
    [
        ("tmp", {"tmp000": 123, "tmp001": 321}, "tmp002"),
        ("empty", {}, "empty000")
    ]
)
def test_get_default_alias(prefix, objects, expected):
    actual = utils.get_default_alias(prefix, objects)
    assert actual == expected


@pytest.mark.parametrize(
    "start, duration, expected_len, raises",
    [
        (None, None, utils.SAMPLE_RATE, False),      # 1.0s duration
        (0.5, 0.5, utils.SAMPLE_RATE // 2, False),    # 0.5s duration
        (0.25, 0.5, utils.SAMPLE_RATE // 2, False),        # 0.5s duration
        (0.75, 0.25, utils.SAMPLE_RATE // 4, False),     # 0.25s duration
        (0.0, 1.0, utils.SAMPLE_RATE, False),        # 1.0s duration
        (0.0, None, utils.SAMPLE_RATE, False),       # 1.0s duration
        (0.9, 0.2, None, ValueError),          # 0.9s + 0.2s = 1.1s > 1.0s, ValueError
        (-0.1, 0.5, None, ValueError),         # negative start, ValueError
        (0.0, -1.0, None, ValueError),         # negative duration, ValueError
    ]
)
def test_truncate_audio(start, duration, expected_len, raises):
    audio = np.random.rand(utils.SAMPLE_RATE)
    if not raises:
        output = utils.truncate_audio(audio, sr=utils.SAMPLE_RATE, start=start, duration=duration)
        assert isinstance(output, np.ndarray)
        assert output.ndim == 1
        assert len(output) == expected_len
    else:
        with pytest.raises(raises):
            utils.truncate_audio(audio, sr=utils.SAMPLE_RATE, start=start, duration=duration)


@pytest.mark.parametrize("n_channels", [1, 2, 3, 4])
def test_audio_to_mono(n_channels: int):
    audio = np.random.rand(n_channels, utils.SAMPLE_RATE)
    mono = utils.audio_to_mono(audio)
    assert mono.ndim == 1


# noinspection PyTypeChecker
def test_validate_audio():
    for bad in ["asdf", 0, set]:
        with pytest.raises(ValueError):
            _ = utils.validate_audio(bad)
    with pytest.raises(TypeError):
        _ = utils.validate_audio(np.array([False, False, True], dtype=bool))
    with pytest.raises(ValueError):
        _ = utils.validate_audio(np.array(0.0))
    with pytest.raises(ValueError):
        _ = utils.validate_audio(np.array([np.nan, np.nan, np.inf, np.nan, 0.0, 1.0]))


@pytest.mark.parametrize(
    "func, kwargs, should_raise, expected_exception",
    [
        (lambda x=1, y=2: x + y, {"x": 5}, False, None),                      # Valid kwarg
        (lambda x=1, y=2: x + y, {"z": 5}, True, AttributeError),            # Invalid kwarg
        (lambda **kwargs: sum(kwargs.values()), {"a": 1}, False, None),     # Accepts arbitrary kwargs
        ("not_a_function", {"x": 1}, True, TypeError),                      # Not a callable
        (lambda x, y: x + y, {}, False, None),                               # No kwargs but valid empty call
        (lambda *, a=1: a, {"a": 2}, False, None),                           # Keyword-only argument
        (lambda a, b: a + b, {"c": 3}, True, AttributeError),               # Invalid kwarg
    ]
)
def test_validate_kwargs(func, kwargs, should_raise, expected_exception):
    if should_raise:
        with pytest.raises(expected_exception):
            utils.validate_kwargs(func, **kwargs)
    else:
        utils.validate_kwargs(func, **kwargs)


@pytest.mark.parametrize(
    "directory,raises",
    [
        (utils.get_project_root() / "tests", False,),
        ("asdfasgsfhdsh", FileNotFoundError,),
        (utils.get_project_root() / "tests/test_utils.py", ValueError,)
    ]
)
def test_list_all_directories(directory, raises):
    if raises:
        with pytest.raises(raises):
            _ = utils.list_all_directories(directory)
    else:
        out = utils.list_all_directories(directory)
        assert len(out) >= 1
        for path in out:
            assert os.path.isdir(path)
            assert isinstance(path, str)


@pytest.mark.parametrize(
    "directory,expected",
    [
        (utils.get_project_root() / "tests", ["femaleSpeech", "music", "meshes"])
    ]
)
def test_list_deepest_directories(directory, expected):
    out = utils.list_deepest_directories(directory)
    assert isinstance(out, list)
    for expect in expected:
        assert any([expect in actual for actual in out])


@pytest.mark.parametrize(
    "directory,expected",
    [
        (utils.get_project_root() / "tests", ["femaleSpeech", "music", "meshes"])
    ]
)
def test_list_innermost_directory_names_unique(directory, expected):
    out = utils.list_innermost_directory_names_unique(directory)
    assert isinstance(out, set)
    for expect in expected:
        assert any([expect in actual for actual in out])


@pytest.mark.parametrize(
    "shape_a,shape_b,raises",
    [
        # Matching shapes
        ((3, 4), (3, 4), False),
        ((5,), (5,), False),
        ((3, 4, 5), (3, 4, 5), False),
        # Non-matching shapes
        ((3, 4), (3, 5), True),
        ((3,), (4,), True),
        ((3, 4, 5), (3, 4, 6), True),
        ((1,), (2, 1), True),
    ]
)
def test_validate_shape(shape_a: tuple[Union[int, None]], shape_b: tuple[Union[int, None]], raises: bool):
    if raises:
        with pytest.raises(ValueError):
            utils.validate_shape(shape_a, shape_b)
    else:
        utils.validate_shape(shape_a, shape_b)
