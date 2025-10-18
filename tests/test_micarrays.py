#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests microphone arrays in audiblelight.micarrays"""

import json

import numpy as np
import pytest
from rlr_audio_propagation import ChannelLayout

from audiblelight import utils
from audiblelight.micarrays import (
    MICARRAY_LIST,
    AmbeoVR,
    Eigenmike32,
    MicArray,
    MonoCapsule,
    sanitize_microphone_input,
)


@pytest.mark.parametrize("micarray", MICARRAY_LIST)
def test_string_attributes(micarray):
    # The class should have all the desired attributes as non-empty strings
    for attr in ["name", "channel_layout_type"]:
        assert hasattr(micarray(), attr)
        assert isinstance(getattr(micarray(), attr), str)
        assert getattr(micarray(), attr) != ""


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
        assert (
            polar.shape
            == cartesian.shape
            == (micarray().n_capsules, 3)
            == (len(micarray()), 3)
        )
        # All azimuth values must be in range [-180, 180]
        assert all(-180 <= p <= 180 for p in polar[:, 0])


@pytest.mark.parametrize("micarray", MICARRAY_LIST)
def test_cartesian_coordinates(micarray):
    # Should have cartesian coordinates as an array type
    assert hasattr(micarray, "coordinates_cartesian")
    cartesian: np.ndarray = getattr(micarray(), "coordinates_cartesian")
    assert isinstance(cartesian, np.ndarray)
    # Everything should have the same shape
    if micarray.channel_layout_type == "mic":
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
    "micarray,center", [(ma, np.array([5.0, 5.0, 5.0])) for ma in MICARRAY_LIST]
)
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
    [
        ("eigenmike32", Eigenmike32),
        ("ambeovr", AmbeoVR),
        ("asdf", ValueError),
        (None, MonoCapsule),
        (123, TypeError),
    ],
)
def test_sanitize_microphone_input(array_name: str, expected: object):
    if issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = sanitize_microphone_input(array_name)
    else:
        assert type(expected) is type(sanitize_microphone_input(array_name))


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "name": "ambeovr",
            "micarray_type": "AmbeoVR",
            "is_spherical": True,
            "channel_layout_type": "mic",
            "n_capsules": 4,
            "capsule_names": ["FLU", "FRD", "BLD", "BRU"],
            "coordinates_absolute": [
                [2.1146615660317107, 0.0029858628742159927, 2.029366448659064],
                [2.1146615660317107, -0.008598696432575392, 2.0178949199320435],
                [2.1030770067249196, 0.0029858628742159927, 2.0178949199320435],
                [2.1030770067249196, -0.008598696432575392, 2.029366448659064],
            ],
            "coordinates_center": [
                2.108869286378315,
                -0.002806416779179699,
                2.023630684295554,
            ],
        }
    ],
)
def test_micarray_from_dict(input_dict):
    out_array = MicArray.from_dict(input_dict)
    assert issubclass(type(out_array), MicArray)
    out_dict = out_array.to_dict()
    for k, v in out_dict.items():
        # will be `pop`d and removed
        if k == "micarray_type":
            continue

        if isinstance(input_dict[k], (np.ndarray, list)) and not isinstance(
            input_dict[k][0], str
        ):
            assert np.isclose(input_dict[k], out_dict[k], atol=utils.SMALL).all()
        else:
            assert input_dict[k] == out_dict[k]


@pytest.mark.parametrize("mictype", MICARRAY_LIST)
def test_magic_methods(mictype):
    instant = mictype()
    instant.set_absolute_coordinates([-0.5, -0.5, 0.5])
    for at in [
        "__len__",
        "__repr__",
        "__str__",
    ]:
        assert hasattr(instant, at)
        _ = getattr(instant, at)
    # Compare equality
    instant2 = mictype.from_dict(instant.to_dict())
    assert instant == instant2


@pytest.mark.parametrize("mictype", MICARRAY_LIST)
def test_channel_layout(mictype):
    micarray = mictype()
    assert hasattr(micarray, "channel_layout")
    assert hasattr(micarray, "channel_layout_type")
    assert isinstance(micarray.channel_layout, ChannelLayout)
    # assert micarray.channel_layout.channel_count == micarray.n_capsules
