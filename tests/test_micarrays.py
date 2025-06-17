#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests microphone arrays in audiblelight.micarrays"""

import pytest
import numpy as np

from audiblelight.micarrays import *


# A list of microphone arrays to test
MICARRAY_LIST = [
    Eigenmike32,
    Eigenmike64,
    AmbeoVR,
    DeccaCuboid,
    Oct3D,
    PCMA3D,
    Cube2L,
    HamasakiSquare
]

@pytest.mark.parametrize("micarray", MICARRAY_LIST)
def test_string_attributes(micarray):
    # The class should have all the desired attributes as non-empty strings
    assert hasattr(micarray(), "name")
    assert isinstance(getattr(micarray(), "name"), str)
    assert getattr(micarray(), "name") != ""

@pytest.mark.parametrize("micarray", MICARRAY_LIST)
def test_polar_coordinates(micarray):
    # Non-spherical mics will not have polar coordinates
    if not micarray.is_spherical:
        with pytest.raises(NotImplementedError):
            _ = micarray().coordinates_polar
    # Spherical mics have both polar and cartesian coordinates
    else:
        assert hasattr(micarray, "coordinates_polar")
        assert hasattr(micarray, "coordinates_cartesian")
        cartesian: np.ndarray = getattr(micarray(), "coordinates_cartesian")
        polar = getattr(micarray(), "coordinates_polar")
        assert isinstance(polar, np.ndarray)
        # Should have the correct number of capsules and be the same shape as the cartesian coordinates
        assert polar.shape == cartesian.shape == (micarray().n_capsules, 3) == (len(micarray()), 3)


@pytest.mark.parametrize("micarray", MICARRAY_LIST)
def test_cartesian_coordinates(micarray):
    # Should have cartesian coordinates as an array type
    assert hasattr(micarray, "coordinates_cartesian")
    cartesian: np.ndarray = getattr(micarray(), "coordinates_cartesian")
    assert isinstance(cartesian, np.ndarray)
    # Everything should have the same shape
    assert cartesian.shape == (micarray().n_capsules, 3) == (len(micarray()), 3)


@pytest.mark.parametrize(
    "spherical,expected_cartesian",
    [
        (
            np.array([[0.0, 90.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0]])
        ),
        (
            np.array([[90.0, 90.0, 1.0]]),
            np.array([[0.0, 1.0, 0.0]])
        ),
        (
            np.array([[180.0, 90.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0]])
        ),
        (
            np.array([[45.0, 60.0, 2.0]]),
            np.array([
                [
                    2.0 * np.sin(np.deg2rad(60.0)) * np.cos(np.deg2rad(45.0)),
                    2.0 * np.sin(np.deg2rad(60.0)) * np.sin(np.deg2rad(45.0)),
                    2.0 * np.cos(np.deg2rad(60.0)),
                ]
            ]),
        ),
    ]
)
def test_polar_to_cartesian(spherical, expected_cartesian):
    # polar -> cartesian
    result = polar_to_cartesian(spherical)
    assert np.allclose(result, expected_cartesian, atol=1e-6)
    # polar -> cartesian -> polar
    result_pol = cartesian_to_polar(result)
    assert np.allclose(spherical, result_pol)


@pytest.mark.parametrize(
    "cartesian,expected_spherical",
    [
        (
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 90.0, 1.0]])
        ),
        (
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[90.0, 90.0, 1.0]])
        ),
        (
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0]])
        ),
        (
            np.array([[-1.0, -1.0, 0.0]]),
            np.array([[225.0, 90.0, np.sqrt(2)]])
        ),
        (
            np.array([[1.0, 1.0, 1.0]]),
            np.array([[45.0, np.rad2deg(np.arccos(1.0 / np.sqrt(3))), np.sqrt(3)]])
        ),
    ]
)
def test_cartesian_to_polar(cartesian, expected_spherical):
    # cartesian -> polar
    result = cartesian_to_polar(cartesian)
    assert np.allclose(result, expected_spherical)
    # cartesian -> polar -> cartesian
    result_cart = polar_to_cartesian(result)
    assert np.allclose(cartesian, result_cart)


@pytest.mark.parametrize("x,y,z", [(45, 150, 9), (90, 90, 5), (0, 0, 1)])
def test_center_coords(x: int, y: int, z: int):
    coords_dict_deg = np.array([x, y, z])
    coords_dict_cart = polar_to_cartesian(coords_dict_deg)
    coords_dict_centered = center_coordinates(coords_dict_cart)
    # Everything should be centered around the mean
    #  As we only passed in one row, this will mean that every coordinate becomes a 0
    assert np.allclose(
        np.mean(coords_dict_centered, axis=0), [0, 0, 0]
    )


@pytest.mark.parametrize("micarray,center", [(ma, np.array([5.0, 5.0, 5.0])) for ma in MICARRAY_LIST])
def test_absolute_coordinates(micarray, center: np.ndarray):
    abs_coords = micarray().coordinates_absolute(center)
    assert abs_coords.shape == micarray().coordinates_cartesian.shape
