#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/utils.py"""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from audiblelight import utils


EXISTING_FILE = os.path.join(utils.get_project_root(), "tests/test_resources/meshes/Oyens.glb")


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
