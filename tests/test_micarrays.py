#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests microphone arrays in audiblelight.micarrays"""

import json

import pytest
import numpy as np

from audiblelight import utils
from audiblelight.micarrays import *


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


@pytest.mark.parametrize("micarray", MICARRAY_LIST)
def test_to_dict(micarray):
    micarray = micarray()
    micarray.set_absolute_coordinates([-0.5, -0.5, -0.5])
    dict_out = micarray.to_dict()
    assert isinstance(dict_out, dict)
    try:
        json.dumps(dict_out)
    except (TypeError, OverflowError):
        pytest.fail("Dictionary not JSON serializable")


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
    result = utils.polar_to_cartesian(spherical)
    assert np.allclose(result, expected_cartesian, atol=1e-6)
    # polar -> cartesian -> polar
    result_pol = utils.cartesian_to_polar(result)
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
    result = utils.cartesian_to_polar(cartesian)
    assert np.allclose(result, expected_spherical)
    # cartesian -> polar -> cartesian
    result_cart = utils.polar_to_cartesian(result)
    assert np.allclose(cartesian, result_cart)


@pytest.mark.parametrize("x,y,z", [(45, 150, 9), (90, 90, 5), (0, 0, 1)])
def test_center_coords(x: int, y: int, z: int):
    coords_dict_deg = np.array([x, y, z])
    coords_dict_cart = utils.polar_to_cartesian(coords_dict_deg)
    coords_dict_centered = utils.center_coordinates(coords_dict_cart)
    # Everything should be centered around the mean
    #  As we only passed in one row, this will mean that every coordinate becomes a 0
    assert np.allclose(
        np.mean(coords_dict_centered, axis=0), [0, 0, 0]
    )


@pytest.mark.parametrize("micarray,center", [(ma, np.array([5.0, 5.0, 5.0])) for ma in MICARRAY_LIST])
def test_absolute_coordinates(micarray, center: np.ndarray):
    ma = micarray()
    with pytest.raises(NotImplementedError):
        _ = ma.coordinates_absolute
    with pytest.raises(NotImplementedError):
        _ = ma.coordinates_center
    # Set up the absolute coordinates
    abs_coords = ma.set_absolute_coordinates(center)
    assert abs_coords.shape == ma.coordinates_cartesian.shape
    try:
        _ = ma.coordinates_absolute
        _ = ma.coordinates_center
    except NotImplementedError:
        pytest.fail()


@pytest.mark.parametrize(
    "array_name,expected",
    [("eigenmike32", Eigenmike32), ("ambeovr", AmbeoVR), ("asdf", ValueError), (None, MonoCapsule), (123, TypeError)]
)
def test_sanitize_microphone_input(array_name: str, expected: object):
    if issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = sanitize_microphone_input(array_name)
    else:
        assert type(expected) == type(sanitize_microphone_input(array_name))


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "name": "ambeovr",
            "micarray_type": "AmbeoVR",
            "is_spherical": True,
            "n_capsules": 4,
            "capsule_names": [
                "FLU",
                "FRD",
                "BLD",
                "BRU"
            ],
            "coordinates_absolute": [
                [
                    2.1146615660317107,
                    0.0029858628742159927,
                    2.029366448659064
                ],
                [
                    2.1146615660317107,
                    -0.008598696432575392,
                    2.0178949199320435
                ],
                [
                    2.1030770067249196,
                    0.0029858628742159927,
                    2.0178949199320435
                ],
                [
                    2.1030770067249196,
                    -0.008598696432575392,
                    2.029366448659064
                ]
            ],
            "coordinates_polar": [
                [
                    45.0,
                    55.0,
                    0.01
                ],
                [
                    315.0,
                    125.0,
                    0.01
                ],
                [
                    135.0,
                    125.0,
                    0.01
                ],
                [
                    225.0,
                    55.0,
                    0.01
                ]
            ],
            "coordinates_center": [
                2.108869286378315,
                -0.002806416779179699,
                2.023630684295554
            ],
            "coordinates_cartesian": [
                [
                    0.005792279653395693,
                    0.005792279653395692,
                    0.005735764363510461
                ],
                [
                    0.00579227965339569,
                    -0.005792279653395693,
                    -0.005735764363510461
                ],
                [
                    -0.005792279653395691,
                    0.005792279653395692,
                    -0.005735764363510461
                ],
                [
                    -0.0057922796533956935,
                    -0.005792279653395692,
                    0.005735764363510461
                ]
            ]
        }
    ]
)
def test_micarray_from_dict(input_dict):
    out_array = MicArray.from_dict(input_dict)
    assert issubclass(type(out_array), MicArray)
    out_dict = out_array.to_dict()
    for k, v in out_dict.items():
        # will be `pop`d and removed
        if k == "micarray_type":
            continue
        assert input_dict[k] == out_dict[k]
