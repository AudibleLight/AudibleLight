#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests microphone arrays in audiblelight.micarrays"""

import json
from itertools import combinations

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
    dynamically_define_micarray,
    sanitize_microphone_input,
)
from tests import utils_tests


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
    "array_name,expected,matches",
    [
        ("eigenmike32", Eigenmike32, None),
        ("ambeovr", AmbeoVR, None),
        ("asdf", ValueError, "Cannot find array asdf: expected one of"),
        (None, MonoCapsule, None),
        (123, TypeError, "Could not parse microphone type"),
        (Eigenmike32, Eigenmike32, None),
        (AmbeoVR(), AmbeoVR, None),
        # Test with custom defined MicArrays: represents a user defining their own array
        (utils_tests.CubeMic(), utils_tests.CubeMic, None),
        (utils_tests.CubeMic, utils_tests.CubeMic, None),
    ],
)
def test_sanitize_microphone_input(
    array_name: str, expected: object, matches: str | None
) -> None:
    if issubclass(expected, Exception):
        with pytest.raises(expected, match=matches):
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
        if k in ["coordinates_cartesian", "coordinates_polar"]:
            continue

        if isinstance(input_dict[k], (np.ndarray, list)) and not isinstance(
            input_dict[k][0], str
        ):
            assert np.isclose(input_dict[k], out_dict[k], atol=utils.SMALL).all()
        else:
            assert input_dict[k] == out_dict[k]

    # Test removing necessary key: will break
    input_dict.pop("micarray_type")
    with pytest.raises(KeyError, match="'micarray_type' key not found in input dict"):
        _ = MicArray.from_dict(input_dict)


@pytest.mark.parametrize("mictype", MICARRAY_LIST + [MicArray])
def test_magic_methods(mictype):
    instant = mictype()

    # Set coordinates: should raise an error for base class
    if instant.name != "":
        instant.set_absolute_coordinates([-0.5, -0.5, 0.5])
    else:
        with pytest.raises(NotImplementedError):
            instant.set_absolute_coordinates([-0.5, -0.5, 0.5])

    # Actually check the magic methods
    assert isinstance(instant.__repr__(), str)
    assert isinstance(instant.__str__(), str)
    assert isinstance(instant.__len__(), int)
    assert isinstance(instant.capsule_names, list)

    # Compare equality
    if instant.name != "":
        instant2 = mictype.from_dict(instant.to_dict())
        assert instant == instant2


@pytest.mark.parametrize("mictype", MICARRAY_LIST)
def test_channel_layout(mictype):
    micarray = mictype()
    assert hasattr(micarray, "channel_layout")
    assert hasattr(micarray, "channel_layout_type")
    assert isinstance(micarray.channel_layout, ChannelLayout)
    # assert micarray.channel_layout.channel_count == micarray.n_capsules


@pytest.mark.parametrize(
    "array_kwargs",
    [
        [
            dict(
                name="array1",
                channel_layout_type="mic",
                coordinates_cartesian=[[0.0, 1.0, 0.5]],
                capsule_names=["left"],
            ),
            dict(
                name="array2",
                channel_layout_type="foa",
                micarray_type="Tester",
                coordinates_cartesian=[[1.0, 1.0, 0.5]],
                capsule_names=["right"],
            ),
            dict(
                name="array3",
                channel_layout_type="binaural",
                coordinates_cartesian=[[0.0, -1.0, 0.5], [0.0, 1.0, 0.5]],
                capsule_names=["lower", "upper"],
            ),
        ]
    ],
)
def test_dynamically_define_micarrays(array_kwargs):
    all_arrays = []

    # Dynamically define every array in the list
    for array in array_kwargs:
        defined = dynamically_define_micarray(**array)()
        defined.set_absolute_coordinates([0.0, 0.0, 0.0])

        # Should be a subclass of the MicArray object
        assert issubclass(type(defined), MicArray)

        # Must serialise to a dictionary and unserialise correctly
        assert isinstance(defined.to_dict(), dict)
        assert MicArray.from_dict(defined.to_dict()) == defined

        # Check attributes set correctly
        if "name" in array.keys():
            assert defined.name == array["name"]
        if "channel_layout_type" in array.keys():
            assert defined.channel_layout_type == array["channel_layout_type"]
        if "coordinates_cartesian" in array.keys():
            assert np.array_equal(
                defined.coordinates_cartesian, array["coordinates_cartesian"]
            )
        if "micarray_type" in array.keys():
            assert defined.__class__.__name__ == array["micarray_type"]

        all_arrays.append(defined)

    # Test that arrays are different
    combined_arrays = combinations(all_arrays, 2)
    for a1, a2 in combined_arrays:
        assert a1 != a2


@pytest.mark.parametrize(
    "array_kwargs,missing_attr",
    [
        (
            dict(
                name="array1",
                channel_layout_type="mic",
            ),
            ["coordinates_cartesian", "coordinates_polar", "capsule_names"],
        )
    ],
)
def test_dynamically_define_micarray_bad(array_kwargs, missing_attr):
    array_out = dynamically_define_micarray(**array_kwargs)()
    for attr_ in missing_attr:
        with pytest.raises(NotImplementedError):
            _ = getattr(array_out, attr_)


@pytest.mark.parametrize(
    "mictype,expected,expected_channels,expected_listeners",
    [
        (
            dynamically_define_micarray(
                name="tester",
                channel_layout_type="binaural",
                coordinates_cartesian=[[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]],
                capsule_names=["left", "right"],
            )(),
            "binaural",
            2,
            1,
        ),
        (AmbeoVR(), "mic", 1, 4),
        (sanitize_microphone_input("foalistener")(), "foa", 4, 1),
        (
            dynamically_define_micarray(
                name="tester",
                channel_layout_type="willbreak",
                coordinates_cartesian=[[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]],
                capsule_names=["left", "right"],
            )(),
            ValueError,
            "Expected 'channel_layout_type' to be one of",
            None,
        ),
    ],
)
def test_parse_channel_layout(mictype, expected, expected_channels, expected_listeners):
    if expected is not ValueError:
        assert mictype.channel_layout_type == expected
        assert mictype.channel_layout.channel_count == expected_channels
        assert mictype.n_listeners == expected_listeners
    else:
        with pytest.raises(expected, match=expected_channels):
            _ = mictype.channel_layout
        with pytest.raises(expected, match=expected_channels):
            _ = mictype.n_listeners
