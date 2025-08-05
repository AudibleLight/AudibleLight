#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/worldstate.py"""

import json
import os
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
from pyroomacoustics.doa.music import MUSIC
from scipy.signal import stft
from trimesh import Scene, Trimesh

from audiblelight import utils
from audiblelight.micarrays import (
    MICARRAY_LIST,
    AmbeoVR,
    MonoCapsule,
    sanitize_microphone_input,
)
from audiblelight.worldstate import Emitter, WorldState, load_mesh, repair_mesh

TEST_RESOURCES = utils.get_project_root() / "tests/test_resources/meshes"
TEST_MESHES = [TEST_RESOURCES / glb for glb in TEST_RESOURCES.glob("*.glb")]


@pytest.mark.parametrize("mesh_fpath", TEST_MESHES)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_repair_mesh(mesh_fpath: str):
    # Load up the mesh
    loaded = load_mesh(mesh_fpath)
    # Make a copy of the mesh
    new_mesh = Trimesh(vertices=loaded.vertices.copy(), faces=loaded.faces.copy())
    # Repair the mesh, in-place
    repair_mesh(new_mesh)
    # Should still expect a mesh object to be returned
    assert isinstance(new_mesh, Trimesh)


@pytest.mark.parametrize("mesh_fpath", TEST_MESHES)
def test_load_mesh_from_fpath(mesh_fpath: str):
    loaded = load_mesh(mesh_fpath)
    assert isinstance(loaded, Trimesh)
    assert loaded.metadata["fpath"] == str(
        mesh_fpath
    )  # need both to be a string, or we'll get TypeError
    assert loaded.units == utils.MESH_UNITS  # units should be in meters
    # If we try to load from a mesh object, should raise an error
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = load_mesh(loaded)


@pytest.mark.parametrize(
    "mesh_fpath,expected", [("iamnotafile", FileNotFoundError), (1234, TypeError)]
)
def test_load_broken_mesh(mesh_fpath: str, expected):
    with pytest.raises(expected):
        load_mesh(mesh_fpath)


@pytest.mark.parametrize(
    "microphone_type,position,alias",
    [
        (None, None, None),  # places 1 mono mic in a random position with default alias
        (None, [-0.5, -0.5, 0.5], None),  # places mono mic in assigned position
        (
            "eigenmike32",
            None,
            "customalias000",
        ),  # places eigenmike32 in random position
        (AmbeoVR, None, "ambeovr000"),  # same but with AmbeoVR
        (
            "eigenmike32",
            [-0.5, -0.5, 0.5],
            "customeigenmike000",
        ),  # places eigenmike32 in assigned position
        (
            "ambeovr",
            [-0.1, -0.1, 0.6],
            "ambeovr000",
        ),  # places AmbeoVR in assigned position
    ],
)
def test_add_microphone(microphone_type, position, alias, oyens_space: WorldState):
    """Test adding a single microphone to the space"""
    # Add the microphones to the space: keep_existing=False ensures we remove previously-added microphones
    oyens_space.add_microphone(microphone_type, position, alias, keep_existing=False)
    assert isinstance(oyens_space.microphones, dict)
    assert (
        len(oyens_space.microphones) == 1
    )  # only addding one microphone with this function
    # Test microphone type
    mic = list(oyens_space.microphones.values())[0]
    if microphone_type is None:
        assert isinstance(mic, MonoCapsule)
    elif isinstance(microphone_type, str):
        assert isinstance(mic, sanitize_microphone_input(microphone_type))
    else:
        assert isinstance(mic, microphone_type)
    # Test aliases
    if alias is not None:
        assert (
            list(oyens_space.microphones.keys())[0] == alias
        )  # alias should be what we expect
        mic = oyens_space.get_microphone(alias)
    else:
        mic = oyens_space.get_microphone("mic000")
    # Should have exactly 1 listener for every microphone capsule
    n_capsules = sum([m.n_capsules for m in oyens_space.microphones.values()])
    assert n_capsules == oyens_space.ctx.get_listener_count()
    valid_mics = [ma().name for ma in MICARRAY_LIST]
    # Microphone array type should be valid
    assert mic.name in valid_mics
    # Microphones should have coordinates assigned to them
    assert mic.coordinates_absolute is not None
    assert mic.coordinates_center is not None
    # Iterate over all capsules
    for capsule in mic.coordinates_absolute:
        # TODO: maybe flaky?
        assert oyens_space._is_point_inside_mesh(capsule)


# noinspection PyTypeChecker
def test_place_invalid_microphones(oyens_space: WorldState):
    # Trying to access IRs before placing anything should raise an error
    with pytest.raises(AttributeError):
        _ = oyens_space.irs
    # Cannot add emitters with invalid input types
    for inp in [-1, [], {}, object, set(), lambda x: x]:
        with pytest.raises(TypeError):
            oyens_space.add_microphone(microphone_type=inp, keep_existing=False)
    # Cannot add mic that is way outside the mesh
    for inp in [[1000.0, 1000.0, 1000.0], [-1000, -1000, -1000]]:
        with pytest.raises(ValueError):
            oyens_space.add_microphone(position=inp, keep_existing=False)
    # Cannot add alias that is already in the dictionary
    oyens_space.add_microphone(alias="tmp_alias")
    with pytest.raises(KeyError):
        oyens_space.add_microphone(alias="tmp_alias", keep_existing=True)


@pytest.mark.parametrize(
    "microphone_types,positions,aliases",
    [
        # Two eigenmikes with default aliases
        (
            ["eigenmike32", "eigenmike32"],
            np.array([[-0.5, -0.5, 0.5], [0.6, 0.4, 0.4]]),
            None,
        ),
        # Eigenmike32 and ambeovr in assigned positions with assigned aliases
        (
            ["eigenmike32", "ambeovr"],
            [[-0.5, -0.5, 0.5], [-0.1, -0.1, 0.6]],
            ["eigenmike000", "ambeovr000"],
        ),
        # Three ambeovrs in random positions with default aliases
        (["ambeovr", "ambeovr", AmbeoVR], None, None),
        # A mono capsule, an ambeovr, and an eigenmike (all specified in different ways) with random positions
        ([None, AmbeoVR, "eigenmike32"], None, ["mono", "ambeo", "eigen"]),
    ],
)
def test_add_microphones(microphone_types, positions, aliases, oyens_space: WorldState):
    # Add microphone types into space
    oyens_space.add_microphones(
        microphone_types, positions, aliases, keep_existing=False, raise_on_error=True
    )
    # Check overall dictionary
    assert len(oyens_space.microphones) == len(microphone_types)
    assert isinstance(oyens_space.microphones, dict)
    # Should have exactly 1 listener for every microphone capsule
    n_capsules = sum([m.n_capsules for m in oyens_space.microphones.values()])
    assert n_capsules == oyens_space.ctx.get_listener_count()
    # Iterate over all microphones
    valid_mics = [ma().name for ma in MICARRAY_LIST]
    for mic in oyens_space.microphones.values():
        # Microphone array type should be valid
        assert mic.name in valid_mics
        # Microphones should have coordinates assigned to them
        assert mic.coordinates_absolute is not None
        assert mic.coordinates_center is not None
        # Iterate over all capsules
        for capsule in mic.coordinates_absolute:
            assert oyens_space._is_point_inside_mesh(capsule)
    # If we've provided aliases, check these are correct
    if aliases is not None:
        assert all(a in oyens_space.microphones.keys() for a in aliases)
    # If we've provided positions, check these are correct
    if positions is not None:
        for expected_pos, actual_mic in zip(
            positions, oyens_space.microphones.values()
        ):
            assert np.array_equal(expected_pos, actual_mic.coordinates_center)


# noinspection PyTypeChecker
def test_add_microphones_invalid_inputs(oyens_space: WorldState):
    # Trying to add non-unique aliases raises an error
    with pytest.raises(ValueError):
        oyens_space.add_microphones(
            None, None, ["ambeovr", "ambeovr"], keep_existing=False
        )
    # Trying to add iterables with different lengths raises an error
    with pytest.raises(ValueError):
        oyens_space.add_microphones(
            ["ambeovr", "ambeovr"], None, ["ambeovr"], keep_existing=False
        )
    # Trying to add microphone outside the mesh
    for pos in [[-1000, -1000, -1000], [1000, 1000, 1000]]:
        with pytest.raises(ValueError):
            oyens_space.add_microphones(
                ["ambeovr"],
                [pos],
                ["ambeovr"],
                keep_existing=False,
                raise_on_error=True,
            )
    # Cannot add alias that is already in the dictionary
    oyens_space.add_microphones(aliases=["tmp_alias"])
    with pytest.raises(KeyError):
        oyens_space.add_microphones(aliases=["tmp_alias"], keep_existing=True)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "test_position,expected",
    [
        (np.array([-0.4, -0.5, 0.5]), False),  # Too close to mic
        (np.array([-0.5, -0.4, 0.5]), False),  # Too close to mic
        (np.array([-0.5, -0.5, 0.4]), False),  # Too close to mic
        (np.array([-0.8, -1.5, 0.2]), False),  # Too close to the surface
        (np.array([-0.1, -0.1, 0.6]), True),  # Fine!
        (np.array([0.5, 0.5, 0.5]), True),  # Also fine
        (np.array([0.5]), ValueError),  # should raise an error with invalid array shape
        (np.array([[0.5, 0.5, 0.5], [-0.4, -0.5, 0.5]]), False),  # 1 invalid, 2 valid
        (np.array([[0.5, 0.5, 0.5], [-0.1, -0.1, 0.6]]), True),  # both valid
        (
            np.array([[0.5], [0.5]]),
            ValueError,
        ),  # should raise an error with invalid array shape
    ],
)
def test_validate_position(
    test_position: np.ndarray, expected: bool, oyens_space: WorldState
):
    """Given a microphone with coordinates [-0.5, -0.5, 0.5], test whether test_position is valid"""
    oyens_space.add_microphone(
        microphone_type="ambeovr", position=[-0.5, -0.5, 0.5], keep_existing=False
    )
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            oyens_space._validate_position(test_position)
    else:
        assert oyens_space._validate_position(test_position) == expected


@pytest.mark.parametrize(
    "position,emitter_alias",
    [
        (None, None),  # Add random emitter with no aliases
        ([-0.1, -0.1, 0.6], "custom_alias"),  # add specific emitter with custom alias
        (np.array([-0.5, -0.5, 0.5]), "custom_alias"),  # position as array, not list
    ],
)
def test_add_emitter(position, emitter_alias, oyens_space: WorldState):
    oyens_space.clear_microphones()
    # Add the emitters in and check that the shape of the resulting array is what we expect
    oyens_space.add_emitter(
        position, emitter_alias, mic=None, keep_existing=False, polar=False
    )
    assert isinstance(oyens_space.emitters, dict)
    assert len(oyens_space.emitters) == 1
    # Get the desired emitter: should be the first element in the list
    src = oyens_space.get_emitter(
        emitter_alias if emitter_alias is not None else "src000", 0
    )
    # Should be an emitter object
    assert isinstance(src, Emitter)
    # Should have all the desired attributes
    if emitter_alias is not None:
        assert src.alias == emitter_alias
    if position is not None:
        assert np.array_equal(src.coordinates_absolute, position)
    # Test output dictionary
    di = src.to_dict()
    assert isinstance(di, dict)
    for k in [
        "alias",
        "coordinates_absolute",
        "coordinates_relative_cartesian",
        "coordinates_relative_polar",
    ]:
        assert k in di.keys()
    # Test output strings
    assert isinstance(repr(src), str)
    assert isinstance(str(src), str)
    assert repr(src) != str(src)


def test_add_emitter_invalid(oyens_space: WorldState):
    # Raise error when no microphone with alias has been added
    with pytest.raises(KeyError):
        oyens_space.add_emitter(
            mic="ambeovr", position=[1000, 1000, 1000], keep_existing=False, polar=False
        )
    # Raise error when trying to add emitter out of bounds
    oyens_space.add_microphone(alias="ambeovr")
    with pytest.raises(ValueError):
        oyens_space.add_emitter(
            mic="ambeovr", position=[1000, 1000, 1000], keep_existing=False, polar=False
        )
    # Cannot add emitter that directly intersects with a microphone
    oyens_space.add_microphone(position=[-0.5, -0.5, 0.5], keep_existing=False)
    with pytest.raises(ValueError):
        oyens_space.add_emitter(
            [-0.5, -0.5, 0.5], polar=False
        )  # same, in absolute terms
    with pytest.raises(ValueError):
        oyens_space.add_emitter(
            [0.0, 0.0, 0.0], mic="mic000", polar=False
        )  # same, in relative terms
    # Must provide a reference microphone when using polar emitters
    with pytest.raises(AssertionError):
        oyens_space.add_emitter([0.0, 0.0, 0.0], polar=True, mic=None)
    # Cannot use random positions with polar = True
    with pytest.raises(AssertionError):
        oyens_space.add_emitter(position=None, polar=True)
    # This emitter is valid, but has no direct path to the microphone
    with pytest.raises(ValueError):
        # emitter is in bedroom 2, microphone is in living room
        oyens_space.add_microphone(
            position=np.array([-1.5, -1.5, 0.7]), alias="tester", keep_existing=False
        )
        oyens_space.add_emitter(
            position=np.array([2.9, -7.0, 0.3]),
            polar=False,
            ensure_direct_path="tester",
            keep_existing=False,
        )


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "inputs,outputs",
    [
        (True, ["tester1", "tester2", "tester3"]),
        ("tester1", ["tester1"]),
        (["tester2", "tester3"], ["tester2", "tester3"]),
        (["tester3", "tester3"], ["tester3"]),  # duplicates removed
        (False, []),
        ("tester4", KeyError),  # not a microphone alias
        (["tester1", "tester2", "tester4"], KeyError),  # contains a missing alias
        (object, TypeError),  # cannot handle this type
        (123, TypeError),  # cannot handle this type
    ],
)
def test_get_microphones_from_alias(inputs, outputs, oyens_space: WorldState):
    oyens_space.add_microphones(
        aliases=["tester1", "tester2", "tester3"], keep_existing=False
    )
    if isinstance(outputs, type) and issubclass(outputs, Exception):
        with pytest.raises(outputs):
            _ = oyens_space._parse_valid_microphone_aliases(inputs)
    else:
        actuals = oyens_space._parse_valid_microphone_aliases(inputs)
        assert sorted(actuals) == outputs


@pytest.mark.parametrize(
    "emitter_position,expected_position",
    [
        # emitter offset 20 cm along +x direction (azimuth=0°, colatitude=90°)
        ([0.0, 90.0, 0.2], [-0.3, -0.5, 0.5]),
        # emitter offset 20 cm along +y direction (azimuth=90°, colatitude=90°)
        (np.array([90.0, 90.0, 0.2]), np.array([-0.5, -0.3, 0.5])),
        # emitter offset 20 cm along -z direction (azimuth=90°, colatitude=180°)
        (np.array([90.0, 180.0, 0.2]), [-0.5, -0.5, 0.3]),
        # emitter directly above the mic, 20 cm along +z (colatitude=0°)
        ([0.0, 0.0, 0.2], [-0.5, -0.5, 0.7]),
        # emitter directly below the mic, 20 cm along -z (colatitude=180°)
        ([0.0, 180.0, 0.2], [-0.5, -0.5, 0.3]),
        # emitter offset 30 cm along +y direction (azimuth=90°, colatitude=90°)
        ([90.0, 90.0, 0.3], [-0.5, -0.2, 0.5]),
        # emitter diagonally down-forward (azimuth=45°, colatitude=135°)
        ([45.0, 135.0, 0.2], [-0.4, -0.4, 0.5 - 0.1414]),
    ],
)
def test_add_polar_emitter(
    emitter_position, expected_position, oyens_space: WorldState
):
    oyens_space.add_microphone(
        keep_existing=False,
        position=[-0.5, -0.5, 0.5],
        microphone_type="monocapsule",
        alias="tester",
    )
    oyens_space.add_emitter(
        position=emitter_position,
        polar=True,
        mic="tester",
        keep_existing=False,
        alias="testsrc",
    )
    assert np.allclose(
        oyens_space.get_emitter("testsrc", 0).coordinates_absolute,
        expected_position,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "position,accept",
    [
        ([0.1, 0.0, 0.0], False),
        (np.array([0.0, 0.1, 0.0]), False),
        ([1000, 1000, 1000], False),
        ([-0.2, 0.2, 0.2], True),
        ([0.2, -0.3, -0.2], True),
    ],
)
def test_add_emitter_relative_to_mic(position, accept: bool, oyens_space: WorldState):
    # Add a microphone to the space
    oyens_space.add_microphone(
        microphone_type="ambeovr",
        position=[-0.5, -0.5, 0.5],
        alias="tester",
        keep_existing=False,
    )
    # Trying to add an emitter that should be rejected
    if not accept:
        with pytest.raises(ValueError):
            oyens_space.add_emitter(
                position=position, mic="tester", keep_existing=False, polar=False
            )
    else:
        oyens_space.add_emitter(
            position=position, mic="tester", keep_existing=False, polar=False
        )
        assert len(oyens_space.emitters) == 1
        src = oyens_space.get_emitter("src000", 0)
        assert isinstance(src, Emitter)
        # coordinates_relative dict should be as expected
        assert np.allclose(
            src.coordinates_relative_cartesian["tester"], position, atol=1e-4
        )
        assert np.allclose(
            src.coordinates_relative_polar["tester"],
            utils.cartesian_to_polar(position),
            atol=1e-4,
        )


@pytest.mark.parametrize(
    "positions,emitter_aliases",
    [
        (np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]), None),
        (
            np.array([[0.5, 0.5, 0.5], [0.6, 0.2, 0.5]]),
            ["custom_alias1", "custom_alias2"],
        ),
        (
            [[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5], [-0.4, -0.5, 0.5]],
            ["custom_alias1", "custom_alias2", "custom_alias3"],
        ),
    ],
)
def test_add_emitters(positions, emitter_aliases, oyens_space: WorldState):
    oyens_space.clear_microphones()
    oyens_space.add_emitters(
        positions, emitter_aliases, keep_existing=False, polar=False
    )
    assert len(oyens_space.emitters) == len(positions)
    if emitter_aliases is not None:
        assert set(oyens_space.emitters.keys()) == set(emitter_aliases)
        # Should have all the other emitters in our relative coords dict
        for emitter_list in oyens_space.emitters.values():
            for emitter in emitter_list:
                assert set(emitter.coordinates_relative_cartesian.keys()) == set(
                    emitter_aliases
                )
                assert set(emitter.coordinates_relative_polar.keys()) == set(
                    emitter_aliases
                )
    for emitter_list in oyens_space.emitters.values():
        for emitter in emitter_list:
            assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)


@pytest.mark.parametrize(
    "emitter_positions,expected_positions",
    [
        # 1. Azimuth = 0°, Colatitude = 90° (x+), and Colatitude = 0° (z+)
        # emitter 1: offset 20 cm along +x; emitter 2: offset 20 cm directly above mic
        ([[0.0, 90.0, 0.2], [0.0, 0.0, 0.2]], [[-0.3, -0.5, 0.5], [-0.5, -0.5, 0.7]]),
        # 2. Azimuth = 90°, Colatitude = 90° (y+), and Azimuth = 270°, Colatitude = 90° (y−)
        # emitter 1: offset 20 cm along +y; emitter 2: offset 20 cm along −y
        (
            [[90.0, 90.0, 0.2], [270.0, 90.0, 0.2]],
            [[-0.5, -0.3, 0.5], [-0.5, -0.7, 0.5]],
        ),
    ],
)
def test_add_polar_emitters(
    emitter_positions, expected_positions, oyens_space: WorldState
):
    oyens_space.add_microphone(
        keep_existing=False,
        position=[-0.5, -0.5, 0.5],
        microphone_type="monocapsule",
        alias="tester",
    )
    oyens_space.add_emitters(
        positions=emitter_positions, polar=True, mics="tester", keep_existing=False
    )
    for emitter_list, expected_position in zip(
        oyens_space.emitters.values(), expected_positions
    ):
        for emitter in emitter_list:
            assert np.allclose(
                emitter.coordinates_absolute, expected_position, atol=1e-4
            )


@pytest.mark.parametrize(
    "test_position,expected",
    [
        (
            np.array([[0.1, 0.0, 0.0], [-0.2, 0.2, 0.2]]),
            (False, True),
        ),  # 1: too close to mic, so skipped, 2: fine
        (
            [[-0.2, 0.2, 0.2], [-0.2, 0.3, 0.2]],
            (True, False),
        ),  # 1: fine, 2: too close to emitter 1, so skipped
        (np.array([[-0.2, 0.2, 0.2], [0.2, -0.3, -0.2]]), (True, True)),  # both fine
    ],
)
def test_add_emitters_relative_to_mic(
    test_position: np.ndarray, expected: tuple[bool], oyens_space: WorldState
):
    # Clear everything out
    oyens_space.clear_microphones()
    oyens_space.clear_emitters()
    oyens_space.add_microphone(
        microphone_type=AmbeoVR,
        position=[-0.5, -0.5, 0.5],
        alias="testmic",
        keep_existing=False,
    )
    # Add the emitters in and check that the shape of the resulting array is what we expect
    #  We set `raise_on_error=False` so we skip over raising an error for invalid emitters
    emit_aliases = [f"test{i}" for i in range(len(test_position))]
    oyens_space.add_emitters(
        positions=test_position,
        mics="testmic",
        keep_existing=False,
        raise_on_error=False,
        polar=False,
        aliases=emit_aliases,
    )
    assert len(oyens_space.emitters) == sum(expected)
    for position, is_added, alias in zip(test_position, expected, emit_aliases):
        if is_added:
            emitter_list = oyens_space[
                alias
            ]  # can also get emitters in this way, too :)
            # Relative position dictionary should be as we expect
            for emitter in emitter_list:
                assert np.allclose(
                    emitter.coordinates_relative_cartesian["testmic"],
                    position,
                    atol=1e-4,
                )
                assert np.allclose(
                    emitter.coordinates_relative_polar["testmic"],
                    utils.cartesian_to_polar(position),
                    atol=1e-4,
                )
                assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)


@pytest.mark.parametrize(
    "n_emitters",
    [i for i in range(1, 10, 2)],
)
def test_add_n_emitters(n_emitters, oyens_space: WorldState):
    oyens_space.add_emitters(n_emitters=n_emitters, keep_existing=False, polar=False)
    assert len(oyens_space.emitters) == n_emitters
    for emitter_list in oyens_space.emitters.values():
        for emitter in emitter_list:
            assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)
            # Should update the relative position dictionary
            assert len(emitter.coordinates_relative_polar) == n_emitters
            assert len(emitter.coordinates_relative_cartesian) == n_emitters
            assert isinstance(emitter.coordinates_absolute, np.ndarray)


@pytest.mark.parametrize(
    "test_position,expected",
    [
        (
            np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]),
            (False, True),
        ),  # 1: too close to mic, 2: fine
        (
            np.array([[0.5, 0.5, 0.5], [0.6, 0.4, 0.5]]),
            (True, False),
        ),  # 1: fine, 2: too close to emitter 1
        ([[-0.1, -0.1, 0.6]], (True,)),
        ([[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5]], (True, True)),
    ],
)
def test_add_emitters_at_specific_position(
    test_position: np.ndarray, expected: tuple[bool], oyens_space: WorldState
):
    oyens_space.add_microphone(
        microphone_type=AmbeoVR, position=[-0.5, -0.5, 0.5], keep_existing=False
    )
    # Add the emitters in and check that the shape of the resulting array is what we expect
    emit_alias = [f"emit{i}" for i in range(len(test_position))]
    oyens_space.add_emitters(
        positions=test_position,
        keep_existing=False,
        raise_on_error=False,
        polar=False,
        aliases=emit_alias,
    )
    assert len(oyens_space.emitters) == sum(expected)
    for position, is_added, alias in zip(test_position, expected, emit_alias):
        if is_added:
            for emitter in oyens_space[alias]:
                assert np.allclose(emitter.coordinates_absolute, position, atol=1e-4)


def test_add_emitters_invalid(oyens_space: WorldState):
    # n_emitters must be a postive integer
    with pytest.raises(AssertionError):
        for inp in ["asdf", [], -1, -100, 0]:
            oyens_space.add_emitters(n_emitters=inp, keep_existing=False, polar=False)
    # Cannot specify both a number of random emitters and positions for them
    with pytest.raises(TypeError):
        oyens_space.add_emitters(positions=[[0, 0, 0]], n_emitters=1, polar=False)
    # Aliases for emitters must be unique
    # with pytest.raises(ValueError):
    #     oyens_space.add_emitters(aliases=["asdf", "asdf"], polar=False)
    # Cannot add emitters that are way outside the mesh
    with pytest.raises(ValueError):
        oyens_space.add_emitters(
            [[1000.0, 1000.0, 1000.0], [-1000, -1000, -1000]],
            keep_existing=False,
            polar=False,
        )


@pytest.mark.parametrize("num_rays", [1, 10, 100])
def test_calculate_weighted_average_ray_length(num_rays: int, oyens_space: WorldState):
    # Get a random valid point inside the mesh
    point = oyens_space.get_random_position()
    result = oyens_space.calculate_weighted_average_ray_length(point, num_rays=num_rays)
    # Validate output is positive float and finite (since rays should hit mesh)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result > 0


@pytest.mark.parametrize("test_num", range(1, 5))
def test_get_random_position(test_num: int, oyens_space: WorldState):
    # For reproducible results
    utils.seed_everything(test_num)
    # Add some microphones and emitters to the space
    for idx_ in range(test_num):
        oyens_space.add_microphone(keep_existing=True)
        oyens_space.add_emitter(keep_existing=True, polar=False)
    # Grab a random position
    random_point = oyens_space.get_random_position()
    # It should be valid (suitable distance from surfaces, inside mesh, away from mics/emitters...)
    assert oyens_space._validate_position(random_point)
    assert random_point.shape == (3,)  # should be a 1D array of XYZ


# Goes (1 mic, 4 emitters), (2 mics, 3 emitters), (3 mics, 2 emitters), (4 mics, 1 emitter)
@pytest.mark.parametrize(
    "n_mics,n_emitters", [(m, s) for m, s in zip(list(range(1, 5))[::-1], range(1, 5))]
)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_simulated_ir(n_mics: int, n_emitters: int, oyens_space: WorldState):
    # For reproducible results
    utils.seed_everything(n_emitters)
    # Add some emitters and microphones to the space
    #  We could use other microphone types, but they're slow to simulate
    oyens_space.add_microphones(
        microphone_types=["ambeovr" for _ in range(n_mics)], keep_existing=False
    )
    oyens_space.add_emitters(n_emitters=n_emitters, polar=False)
    # Grab the IRs: we should have one array for every microphone
    oyens_space.simulate()
    assert isinstance(oyens_space.irs, dict)
    simulated_irs = list(oyens_space.irs.values())
    assert len(simulated_irs) == n_mics
    # Iterate over each individual microphone
    total_capsules = 0
    for mic in oyens_space.microphones.values():
        # Grab the shape of the IRs for this microphone
        actual_capsules, actual_emitters, actual_samples = mic.irs.shape
        # We should have the expected number of emitters, capsules, and samples
        assert actual_emitters == n_emitters
        assert actual_capsules == mic.n_capsules
        assert actual_samples >= 1  # difficult to test number of samples
        total_capsules += actual_capsules
    # IRs for all microphones should have same number of emitters and samples
    _, mic_1_emitters, mic_1_samples = oyens_space.get_microphone("mic000").irs.shape
    assert all(
        [m.irs.shape[1] == mic_1_emitters for m in oyens_space.microphones.values()]
    )
    assert all(
        [m.irs.shape[2] == mic_1_samples for m in oyens_space.microphones.values()]
    )
    # Number of capsules should be the same as the "raw" results of the raytracing engine
    assert total_capsules == oyens_space.ctx.get_audio().shape[0]


def test_create_plot(oyens_space):
    # Add some microphones and emitters
    oyens_space.add_microphone(keep_existing=False)
    oyens_space.add_emitter(polar=False)
    # Create the plot
    fig = oyens_space.create_plot()
    assert isinstance(fig, plt.Figure)
    # Should have two axes for the two views
    assert len(fig.get_axes()) == 2


def test_create_scene(oyens_space):
    # Add some microphones and emitters
    oyens_space.add_microphone(keep_existing=False)
    oyens_space.add_emitter(polar=False)
    # Create the scene
    scene = oyens_space.create_scene()
    assert isinstance(scene, Scene)
    # Should have more geometry than the "raw" scene (without adding spheres for capsules/emitters)
    assert len(scene.geometry) > len(oyens_space.mesh.scene().geometry)


@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_save_wavs(oyens_space: WorldState):
    # Add some microphones and emitters
    oyens_space.add_microphone(
        microphone_type="ambeovr", keep_existing=False
    )  # just adds an ambeovr mic in a random plcae
    oyens_space.add_emitter(polar=False)
    # Run the simulation
    oyens_space.simulate()
    # Dump the IRs to a temporary directory
    with TemporaryDirectory() as tmp:
        oyens_space.save_irs_to_wav(tmp)
        # We have 1 microphone with 4 capsules and 1 sound emitter
        #  We should have saved a WAV file for each of these
        for caps_idx in range(4):
            # The WAV file should exist
            fp = os.path.join(tmp, f"mic000_capsule00{caps_idx}_emitter000.wav")
            assert os.path.exists(fp)
            # Load up the WAV file in librosa and get the number of samples
            y, _ = sf.read(
                fp,
            )
            # Compare to the original IR
            x = oyens_space.irs["mic000"][caps_idx][0]
            assert np.allclose(y, x, atol=1e-4)
    # Temporary directory is implicitly cleaned up


@pytest.mark.parametrize(
    "point_a,point_b,expected_result",
    [
        # Point A in bedroom 1, point B in bedroom 2: should have no direct line
        (np.array([-1.5, -1.5, 0.7]), np.array([2.9, -7.0, 0.3]), False),
        # Point A and B both in living room, should have a direct line
        ([2.5, 0.0, 0.5], [2.4, -1.0, 0.7], True),
        # Point A in living room, point B in bedroom 2
        (np.array([2.5, 0.0, 0.5]), [2.9, -7.0, 0.3], False),
    ],
)
def test_path_between_points(
    point_a: np.ndarray,
    point_b: np.ndarray,
    expected_result: bool,
    oyens_space: WorldState,
):
    """Tests function for ensuring a direct path exists between two points inside a mesh"""
    # Go "both ways" for this function: result should be identical
    result1 = oyens_space.path_exists_between_points(point_a, point_b)
    result2 = oyens_space.path_exists_between_points(point_b, point_a)
    assert result1 == expected_result == result2


@pytest.mark.parametrize(
    "microphone,emitters,actual_doa",
    [
        # Test case 1: two emitters at 90 and 270 degree angles from the mic
        (
            [-1.5, -1.5, 0.7],  # mic placed in bedroom 1
            [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
            [90, 270],
        ),
        # Test case 2: two emitters at 0 and 180 degree angles from the mic
        (
            [2.9, -7.0, 0.3],  # mic placed in bedroom 2
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [0, 180],
        ),
        # Test case 3: single sound emitter at a 45-degree angle
        (
            [2.5, -1.0, 0.5],  # mic placed in living room
            [[1.0, 1.0, 0.0]],
            [
                45,
            ],
        ),
    ],
)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_simulated_doa_with_music(
    microphone: list, emitters: list, actual_doa: list[int], oyens_space: WorldState
):
    """
    Tests DOA of simulated sound emitters and microphones with MUSIC algorithm.

    Places an Eigenmike32, simulates sound emitters, runs MUSIC, checks that estimated DOA is near to actual DOA
    """
    # Add the microphones and simulate the space
    oyens_space.add_microphone(
        microphone_type="eigenmike32",
        position=microphone,
        keep_existing=False,
        alias="tester",
    )
    oyens_space.add_emitters(
        positions=emitters, mics="tester", keep_existing=False, polar=False
    )
    oyens_space.simulate()
    # TODO: in the future we should use simulated sound emitters, not the IRs
    output = oyens_space.irs

    # Create the MUSIC object
    L = oyens_space.get_microphone(
        "tester"
    ).coordinates_absolute.T  # coordinates of our capsules for the eigenmike
    fs = int(oyens_space.ctx.config.sample_rate)
    nfft = 1024
    num_emitters = len(oyens_space.emitters)  # number of sound emitters we've added
    assert num_emitters == len(actual_doa) == len(emitters)  # sanity check everything
    music = MUSIC(
        L=L,
        fs=fs,
        nfft=nfft,
        azimuth=np.deg2rad(np.arange(360)),
        num_sources=num_emitters,
    )

    # Iterating over all of our sound emitters
    for doa_deg_true, emitter_idx in zip(actual_doa, range(num_emitters)):
        # Get the IRs for this emitter: shape (N_capsules=32, 1=mono, N_samples)
        signals = np.vstack([m[:, emitter_idx, :] for m in output.values()])
        # Iterate over each individual IR (one per capsule: shape = 1, N_samples) and compute the STFT
        #  Stacked shape is (N_capsules, (N_fft / 2) + 1, N_frames)
        stft_signals = np.stack(
            [
                stft(cs, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
                for cs in signals
            ]
        )
        # Sanity check the returned shape
        x, y, _ = stft_signals.shape
        assert x == oyens_space.get_microphone("tester").n_capsules
        assert y == (nfft / 2) + 1
        # Run the music algorithm and get the predicted DOA
        music.locate_sources(stft_signals)
        doa_deg_pred = np.rad2deg(music.azimuth_recon[0])
        # Check that the predicted DOA is within a window of tolerance
        diff = abs(doa_deg_pred - doa_deg_true) % 360
        diff = min(diff, 360 - diff)  # smallest distance between angles
        assert diff <= 30


@pytest.mark.parametrize(
    "closemic_position,farmic_position,emitter_position",
    [
        # Testing "length-wise" in the room
        (
            [1.0, -9.5, 0.7],
            [1.0, 0.5, 0.7],
            [0.0, 0.5, 0.0],
        ),
        # Testing "width-wise" in the room
        (
            [0.5, -3.5, 0.7],
            [5.5, -3.5, 0.7],
            [0.5, 0.0, 0.0],
        ),
        # Testing "vertical-wise" in the room
        (
            [0.5, -3.5, 0.3],
            [0.5, -3.5, 0.9],
            [0.5, 0.0, 0.3],
        ),
    ],
)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_simulated_sound_distance(
    closemic_position: list, farmic_position: list, emitter_position: list, oyens_space
):
    """
    Tests distance of simulated sound emitters and microphones.

    Places a emitter and two AmbeoVR microphones near and far, then checks that the sound hits the close mic before far
    """

    oyens_space.clear_microphones()
    oyens_space.clear_emitters()
    # Add the microphones and simulate the space
    oyens_space.add_microphones(
        microphone_types=["ambeovr", AmbeoVR],
        positions=[closemic_position, farmic_position],
        aliases=["closemic", "farmic"],
        keep_existing=False,
    )
    oyens_space.add_emitter(
        emitter_position, mic="closemic", keep_existing=False, polar=False
    )
    oyens_space.simulate()
    irs = oyens_space.irs
    # Shape of the IRs should be as expected
    assert len(irs) == 2
    # Get the IDX of the sample at which the sound hits both microphones
    arrival_close = min(np.flatnonzero(irs["closemic"]))
    arrival_far = min(np.flatnonzero(irs["farmic"]))
    # Should hit the closer mic before the further mic
    assert arrival_close < arrival_far


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (dict(sample_rate=22050, global_volume=0.5), None),
        (dict(will_raise="an_error", sample_rate=595959), AttributeError),
    ],
)
def test_config_parse(cfg, expected):
    if expected is None:
        space = WorldState(mesh=str(TEST_MESHES[-1]), rlr_kwargs=cfg)
        for ke, val in cfg.items():
            assert getattr(space.ctx.config, ke) == val

    else:
        with pytest.raises(expected):
            _ = WorldState(mesh=str(TEST_MESHES[-1]), rlr_kwargs=cfg)


def test_to_dict(oyens_space: WorldState):
    oyens_space.add_microphone(
        microphone_type="ambeovr", alias="tester_mic", keep_existing=False
    )
    oyens_space.add_emitter(alias="tester_emitter", polar=False, keep_existing=False)
    dict_out = oyens_space.to_dict()
    # Dictionary should have required keys
    assert isinstance(dict_out, dict)
    assert "tester_mic" in dict_out["microphones"]
    assert "tester_emitter" in dict_out["emitters"]
    # Object must be JSON serializable
    try:
        json.dumps(dict_out)
    except (TypeError, OverflowError):
        pytest.fail("Dictionary not JSON serializable")


@pytest.mark.parametrize(
    "position,polar,ensure_direct_path,passes",
    [
        # Test with Polar and Cartesian coordinate systems
        (np.array([0.0, 90.0, 0.2]), True, True, True),
        (np.array([0.5, 0.5, -0.5]), False, True, True),
        (np.array([1000, 1000, 1000]), False, False, False),
    ],
)
def test_add_microphone_and_emitter(
    position, polar, ensure_direct_path, passes, oyens_space
):
    if not passes:
        with pytest.raises(
            ValueError, match="Could not place microphone and emitter with specified"
        ):
            oyens_space.add_microphone_and_emitter(
                position=position,
                polar=polar,
                mic_alias="main_mic",
                emitter_alias="test_emitter",
                keep_existing_mics=False,
                keep_existing_emitters=False,
                ensure_direct_path=ensure_direct_path,
                max_place_attempts=10,
            )

    else:
        oyens_space.add_microphone_and_emitter(
            position=position,
            polar=polar,
            mic_alias="main_mic",
            emitter_alias="test_emitter",
            keep_existing_mics=False,
            keep_existing_emitters=False,
            ensure_direct_path=ensure_direct_path,
        )

        # Get the emitter and its relative position to the microphone, whether Polar or Cartesian
        emitter = oyens_space.get_emitter("test_emitter")
        if polar:
            placed_at = emitter.coordinates_relative_polar["main_mic"]
        else:
            placed_at = emitter.coordinates_relative_cartesian["main_mic"]
        if placed_at.shape[0] == 1:
            placed_at = placed_at[0]

        # Should be equivalent with what we passed in
        assert np.allclose(placed_at, position, atol=1e-4)
        # Should be a direct path between the emitter and mic if required
        if ensure_direct_path:
            mic = oyens_space.get_microphone("main_mic")
            assert oyens_space.path_exists_between_points(
                emitter.coordinates_absolute, mic.coordinates_center
            )


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "alias": "event000",
            "coordinates_absolute": [
                3.0982881830339233,
                1.5014361786336288,
                0.8912208770477812,
            ],
            "coordinates_relative_cartesian": {
                "event000": [0.0, 0.0, 0.0],
                "mic000": [0.9894188966556081, 1.5042425954128085, -1.1324098072477726],
                "event001": [
                    [-2.1827582040439326, 3.9942395296395388, 0.23392301334289844]
                ],
                "event002": [
                    [2.135124754128098, 10.294191364884359, -1.0464769708531603]
                ],
                "event003": [
                    [-2.339201127698156, 5.061738730187584, 0.31247350066640633]
                ],
                "event004": [
                    [-0.5173651662307757, 2.3764442382908477, -1.1245813799488733]
                ],
                "event005": [
                    [-0.3446971684637967, 5.411444361516849, -0.3506105206062271]
                ],
                "event006": [
                    [0.5728291268425831, 3.992367357964234, 0.5760568028279776]
                ],
                "event007": [
                    [-1.935785331690945, 3.897029262984338, 0.11192117203554663]
                ],
                "event008": [
                    [-1.2177713250483606, 3.2427999948380837, -0.5478559187089469]
                ],
            },
            "coordinates_relative_polar": {
                "event000": [0.0, 0.0, 0.0],
                "mic000": [[56.66499224846307, 122.16792270266339, 2.1269808439345197]],
                "event001": [
                    [118.65556946856991, 87.05804686855372, 4.557751942967893]
                ],
                "event002": [[78.2823880695226, 95.68441435486638, 10.565237698370836]],
                "event003": [[114.8032598149392, 86.79262228291297, 5.584863523589816]],
                "event004": [
                    [102.2819665837409, 114.81532051865209, 2.679521825449921]
                ],
                "event005": [[93.6446918064515, 93.69956752540523, 5.4337348070088245]],
                "event006": [[81.83486747592731, 81.87159898912284, 4.074183570923376]],
                "event007": [[116.41513438477949, 88.5266131787202, 4.352772481499106]],
                "event008": [[110.58270442150788, 98.98750085799102, 3.50697375443506]],
            },
        }
    ],
)
def test_emitter_from_dict(input_dict):
    out_array = Emitter.from_dict(input_dict)
    assert isinstance(out_array, Emitter)
    out_dict = out_array.to_dict()
    for k, v in out_dict.items():
        assert input_dict[k] == out_dict[k]


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "emitters": {
                "event000": [
                    {
                        "alias": "event000",
                        "coordinates_absolute": [
                            1.6805776963343382,
                            -1.476469978007973,
                            0.8649387688508772,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [0.0, 0.0, 0.0],
                            "mic000": [
                                -3.0445821642747593,
                                -0.7542294127898099,
                                -0.08742983838435414,
                            ],
                            "event001": [
                                [
                                    -0.7221454052769642,
                                    -0.39138970586448174,
                                    -0.9117501248145228,
                                ]
                            ],
                            "event002": [
                                [
                                    -3.5790483309376606,
                                    2.1141422644934034,
                                    -0.48439159269652077,
                                ]
                            ],
                            "event003": [
                                [
                                    -2.1255183066592744,
                                    0.3212464383417508,
                                    0.07981795044351703,
                                ]
                            ],
                            "event004": [
                                [
                                    -3.8517361137018513,
                                    0.880967841525452,
                                    0.021567107998987645,
                                ]
                            ],
                            "event005": [
                                [
                                    0.9543162216796,
                                    -1.5495116954979764,
                                    -1.3639200198196155,
                                ]
                            ],
                            "event006": [
                                [
                                    -2.170100977664368,
                                    2.8309402657157987,
                                    -0.9856614481105637,
                                ]
                            ],
                            "event007": [
                                [
                                    -2.788401803689853,
                                    -0.560675690545505,
                                    -0.2590120118828163,
                                ]
                            ],
                            "event008": [
                                [
                                    0.35517490949911235,
                                    -0.35103848531416126,
                                    0.061847316819019005,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    193.9136801345512,
                                    91.59664696420649,
                                    3.137831502610864,
                                ]
                            ],
                            "event001": [
                                [
                                    208.4568891173388,
                                    137.9845426345415,
                                    1.227178951220487,
                                ]
                            ],
                            "event002": [
                                [149.4296722758364, 96.6466576325053, 4.18495157496237]
                            ],
                            "event003": [
                                [
                                    171.40548112609628,
                                    87.8735534535374,
                                    2.151138826599523,
                                ]
                            ],
                            "event004": [
                                [
                                    167.11692640206846,
                                    89.68726153302713,
                                    3.951258099332983,
                                ]
                            ],
                            "event005": [
                                [
                                    301.62816414607545,
                                    126.8510130228398,
                                    2.2741995879672876,
                                ]
                            ],
                            "event006": [
                                [
                                    127.47252665895243,
                                    105.44691683054752,
                                    3.700687710629241,
                                ]
                            ],
                            "event007": [
                                [
                                    191.3691007241264,
                                    95.20336535325967,
                                    2.8559812798912194,
                                ]
                            ],
                            "event008": [
                                [
                                    315.3355882791931,
                                    82.9399263903941,
                                    0.5031921353787988,
                                ]
                            ],
                        },
                    }
                ],
                "event001": [
                    {
                        "alias": "event001",
                        "coordinates_absolute": [
                            2.4027231016113024,
                            -1.0850802721434913,
                            1.7766888936654,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    0.7221454052769642,
                                    0.39138970586448174,
                                    0.9117501248145228,
                                ]
                            ],
                            "event001": [0.0, 0.0, 0.0],
                            "mic000": [
                                -2.322436758997795,
                                -0.3628397069253282,
                                0.8243202864301686,
                            ],
                            "event002": [
                                [
                                    -2.8569029256606964,
                                    2.505531970357885,
                                    0.427358532118002,
                                ]
                            ],
                            "event003": [
                                [
                                    -1.4033729013823102,
                                    0.7126361442062326,
                                    0.9915680752580398,
                                ]
                            ],
                            "event004": [
                                [
                                    -3.129590708424887,
                                    1.2723575473899338,
                                    0.9333172328135104,
                                ]
                            ],
                            "event005": [
                                [
                                    1.6764616269565642,
                                    -1.1581219896334947,
                                    -0.45216989500509275,
                                ]
                            ],
                            "event006": [
                                [
                                    -1.4479555723874036,
                                    3.2223299715802804,
                                    -0.07391132329604089,
                                ]
                            ],
                            "event007": [
                                [
                                    -2.0662563984128886,
                                    -0.1692859846810233,
                                    0.6527381129317065,
                                ]
                            ],
                            "event008": [
                                [
                                    1.0773203147760766,
                                    0.04035122055032048,
                                    0.9735974416335418,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    28.456889117338775,
                                    42.01545736545852,
                                    1.227178951220487,
                                ]
                            ],
                            "event001": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    188.8796708785771,
                                    70.67506659079316,
                                    2.490957463925499,
                                ]
                            ],
                            "event002": [
                                [
                                    138.74891911398177,
                                    83.58322816176144,
                                    3.8239011619167362,
                                ]
                            ],
                            "event003": [
                                [
                                    153.07842990123382,
                                    57.7895376351471,
                                    1.8602454198933709,
                                ]
                            ],
                            "event004": [
                                [
                                    157.87544678598692,
                                    74.55641776177832,
                                    3.5048983990589773,
                                ]
                            ],
                            "event005": [
                                [
                                    325.3627676629875,
                                    102.51198599001792,
                                    2.0871578147038643,
                                ]
                            ],
                            "event006": [
                                [
                                    114.19679075905682,
                                    91.19856946363518,
                                    3.5334754377331783,
                                ]
                            ],
                            "event007": [
                                [
                                    184.6837157454827,
                                    72.52345326967867,
                                    2.173508751458174,
                                ]
                            ],
                            "event008": [
                                [
                                    2.1450207275691073,
                                    47.91518098783366,
                                    1.452631839106008,
                                ]
                            ],
                        },
                    }
                ],
                "event002": [
                    {
                        "alias": "event002",
                        "coordinates_absolute": [
                            5.259626027271999,
                            -3.5906122425013764,
                            1.349330361547398,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    3.5790483309376606,
                                    -2.1141422644934034,
                                    0.48439159269652077,
                                ]
                            ],
                            "event001": [
                                [
                                    2.8569029256606964,
                                    -2.505531970357885,
                                    -0.427358532118002,
                                ]
                            ],
                            "event002": [0.0, 0.0, 0.0],
                            "mic000": [
                                0.5344661666629014,
                                -2.8683716772832133,
                                0.3969617543121666,
                            ],
                            "event003": [
                                [
                                    1.4535300242783862,
                                    -1.7928958261516525,
                                    0.5642095431400378,
                                ]
                            ],
                            "event004": [
                                [
                                    -0.2726877827641907,
                                    -1.2331744229679513,
                                    0.5059587006955084,
                                ]
                            ],
                            "event005": [
                                [
                                    4.533364552617261,
                                    -3.6636539599913798,
                                    -0.8795284271230948,
                                ]
                            ],
                            "event006": [
                                [
                                    1.4089473532732928,
                                    0.7167980012223953,
                                    -0.5012698554140429,
                                ]
                            ],
                            "event007": [
                                [
                                    0.7906465272478078,
                                    -2.6748179550389084,
                                    0.22537958081370446,
                                ]
                            ],
                            "event008": [
                                [
                                    3.934223240436773,
                                    -2.4651807498075646,
                                    0.5462389095155398,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [329.4296722758364, 83.35334236749472, 4.18495157496237]
                            ],
                            "event001": [
                                [
                                    318.74891911398174,
                                    96.41677183823856,
                                    3.8239011619167362,
                                ]
                            ],
                            "event002": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    280.55492993323514,
                                    82.25241603807378,
                                    2.9446203145285885,
                                ]
                            ],
                            "event003": [
                                [
                                    309.03224853363116,
                                    76.26343537835518,
                                    2.376038169617869,
                                ]
                            ],
                            "event004": [
                                [
                                    257.53102147323443,
                                    68.16836017163281,
                                    1.360541065584085,
                                ]
                            ],
                            "event005": [
                                [
                                    321.0564717734398,
                                    98.58097377063616,
                                    5.894686145979685,
                                ]
                            ],
                            "event006": [
                                [
                                    26.96458822145028,
                                    107.59375007924929,
                                    1.6583737476211609,
                                ]
                            ],
                            "event007": [
                                [
                                    286.4670950671497,
                                    85.38032741736419,
                                    2.798315382349764,
                                ]
                            ],
                            "event008": [
                                [327.928778799393, 83.2897767601882, 4.674784014377919]
                            ],
                        },
                    }
                ],
                "event003": [
                    {
                        "alias": "event003",
                        "coordinates_absolute": [
                            3.8060960029936126,
                            -1.797716416349724,
                            0.7851208184073601,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    2.1255183066592744,
                                    -0.3212464383417508,
                                    -0.07981795044351703,
                                ]
                            ],
                            "event001": [
                                [
                                    1.4033729013823102,
                                    -0.7126361442062326,
                                    -0.9915680752580398,
                                ]
                            ],
                            "event002": [
                                [
                                    -1.4535300242783862,
                                    1.7928958261516525,
                                    -0.5642095431400378,
                                ]
                            ],
                            "event003": [0.0, 0.0, 0.0],
                            "mic000": [
                                -0.9190638576154848,
                                -1.0754758511315607,
                                -0.16724778882787117,
                            ],
                            "event004": [
                                [
                                    -1.7262178070425769,
                                    0.5597214031837012,
                                    -0.05825084244452938,
                                ]
                            ],
                            "event005": [
                                [
                                    3.0798345283388744,
                                    -1.8707581338397272,
                                    -1.4437379702631326,
                                ]
                            ],
                            "event006": [
                                [
                                    -0.04458267100509339,
                                    2.509693827374048,
                                    -1.0654793985540807,
                                ]
                            ],
                            "event007": [
                                [
                                    -0.6628834970305784,
                                    -0.8819221288872559,
                                    -0.33882996232633333,
                                ]
                            ],
                            "event008": [
                                [
                                    2.480693216158387,
                                    -0.6722849236559121,
                                    -0.017970633624498022,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    351.4054811260963,
                                    92.12644654646262,
                                    2.151138826599523,
                                ]
                            ],
                            "event001": [
                                [
                                    333.0784299012338,
                                    122.2104623648529,
                                    1.8602454198933709,
                                ]
                            ],
                            "event002": [
                                [
                                    129.03224853363113,
                                    103.73656462164482,
                                    2.376038169617869,
                                ]
                            ],
                            "event003": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    229.48396380566942,
                                    96.74237415222201,
                                    1.4245344866341507,
                                ]
                            ],
                            "event004": [
                                [
                                    162.03485529569656,
                                    91.83853640828544,
                                    1.8156291271012128,
                                ]
                            ],
                            "event005": [
                                [
                                    328.72459891594906,
                                    111.83352146677213,
                                    3.8819448790593802,
                                ]
                            ],
                            "event006": [
                                [
                                    91.01770591173609,
                                    113.00018372256368,
                                    2.726865796194692,
                                ]
                            ],
                            "event007": [
                                [
                                    233.07019684421743,
                                    107.07249262639269,
                                    1.154126039662299,
                                ]
                            ],
                            "event008": [
                                [
                                    344.83666849227376,
                                    90.40060470231626,
                                    2.5702390540457793,
                                ]
                            ],
                        },
                    }
                ],
                "event004": [
                    {
                        "alias": "event004",
                        "coordinates_absolute": [
                            5.5323138100361895,
                            -2.357437819533425,
                            0.8433716608518895,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    3.8517361137018513,
                                    -0.880967841525452,
                                    -0.021567107998987645,
                                ]
                            ],
                            "event001": [
                                [
                                    3.129590708424887,
                                    -1.2723575473899338,
                                    -0.9333172328135104,
                                ]
                            ],
                            "event002": [
                                [
                                    0.2726877827641907,
                                    1.2331744229679513,
                                    -0.5059587006955084,
                                ]
                            ],
                            "event003": [
                                [
                                    1.7262178070425769,
                                    -0.5597214031837012,
                                    0.05825084244452938,
                                ]
                            ],
                            "event004": [0.0, 0.0, 0.0],
                            "mic000": [
                                0.8071539494270921,
                                -1.635197254315262,
                                -0.10899694638334179,
                            ],
                            "event005": [
                                [
                                    4.806052335381452,
                                    -2.4304795370234284,
                                    -1.3854871278186032,
                                ]
                            ],
                            "event006": [
                                [
                                    1.6816351360374835,
                                    1.9499724241903467,
                                    -1.0072285561095513,
                                ]
                            ],
                            "event007": [
                                [
                                    1.0633343100119985,
                                    -1.441643532070957,
                                    -0.28057911988180395,
                                ]
                            ],
                            "event008": [
                                [
                                    4.206911023200964,
                                    -1.2320063268396133,
                                    0.04028020882003136,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    347.11692640206843,
                                    90.31273846697287,
                                    3.951258099332983,
                                ]
                            ],
                            "event001": [
                                [
                                    337.8754467859869,
                                    105.44358223822168,
                                    3.5048983990589773,
                                ]
                            ],
                            "event002": [
                                [
                                    77.53102147323445,
                                    111.83163982836719,
                                    1.360541065584085,
                                ]
                            ],
                            "event003": [
                                [
                                    342.03485529569656,
                                    88.16146359171456,
                                    1.8156291271012128,
                                ]
                            ],
                            "event004": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    296.2715264167021,
                                    93.42058779915638,
                                    1.8268135900843352,
                                ]
                            ],
                            "event005": [
                                [
                                    333.1737173015784,
                                    104.42678280748758,
                                    5.561020087328059,
                                ]
                            ],
                            "event006": [
                                [
                                    49.2258711086755,
                                    111.36377889745727,
                                    2.764922883210509,
                                ]
                            ],
                            "event007": [
                                [
                                    306.4119202554324,
                                    98.90179977929063,
                                    1.8132127759654588,
                                ]
                            ],
                            "event008": [
                                [
                                    343.6771747356868,
                                    89.47353281492435,
                                    4.383784032285943,
                                ]
                            ],
                        },
                    }
                ],
                "event005": [
                    {
                        "alias": "event005",
                        "coordinates_absolute": [
                            0.7262614746547382,
                            0.07304171749000332,
                            2.2288587886704927,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    -0.9543162216796,
                                    1.5495116954979764,
                                    1.3639200198196155,
                                ]
                            ],
                            "event001": [
                                [
                                    -1.6764616269565642,
                                    1.1581219896334947,
                                    0.45216989500509275,
                                ]
                            ],
                            "event002": [
                                [
                                    -4.533364552617261,
                                    3.6636539599913798,
                                    0.8795284271230948,
                                ]
                            ],
                            "event003": [
                                [
                                    -3.0798345283388744,
                                    1.8707581338397272,
                                    1.4437379702631326,
                                ]
                            ],
                            "event004": [
                                [
                                    -4.806052335381452,
                                    2.4304795370234284,
                                    1.3854871278186032,
                                ]
                            ],
                            "event005": [0.0, 0.0, 0.0],
                            "mic000": [
                                -3.998898385954359,
                                0.7952822827081665,
                                1.2764901814352614,
                            ],
                            "event006": [
                                [
                                    -3.1244171993439678,
                                    4.380451961213775,
                                    0.37825857170905186,
                                ]
                            ],
                            "event007": [
                                [
                                    -3.7427180253694528,
                                    0.9888360049524714,
                                    1.1049080079367992,
                                ]
                            ],
                            "event008": [
                                [
                                    -0.5991413121804876,
                                    1.1984732101838151,
                                    1.4257673366386345,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    121.62816414607543,
                                    53.148986977160206,
                                    2.2741995879672876,
                                ]
                            ],
                            "event001": [
                                [
                                    145.36276766298747,
                                    77.4880140099821,
                                    2.0871578147038643,
                                ]
                            ],
                            "event002": [
                                [
                                    141.0564717734398,
                                    81.41902622936384,
                                    5.894686145979685,
                                ]
                            ],
                            "event003": [
                                [
                                    148.72459891594906,
                                    68.16647853322789,
                                    3.8819448790593802,
                                ]
                            ],
                            "event004": [
                                [
                                    153.17371730157842,
                                    75.57321719251243,
                                    5.561020087328059,
                                ]
                            ],
                            "event005": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    168.75204069113792,
                                    72.61575046983175,
                                    4.272363443537841,
                                ]
                            ],
                            "event006": [
                                [
                                    125.49888323648283,
                                    85.97866156382347,
                                    5.393831825996189,
                                ]
                            ],
                            "event007": [
                                [
                                    165.20046867522083,
                                    74.07010017648396,
                                    4.025736773327169,
                                ]
                            ],
                            "event008": [
                                [
                                    116.5614065030084,
                                    43.22148631815501,
                                    1.9565584186819676,
                                ]
                            ],
                        },
                    }
                ],
                "event006": [
                    {
                        "alias": "event006",
                        "coordinates_absolute": [
                            3.850678673998706,
                            -4.307410243723772,
                            1.8506002169614408,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    2.170100977664368,
                                    -2.8309402657157987,
                                    0.9856614481105637,
                                ]
                            ],
                            "event001": [
                                [
                                    1.4479555723874036,
                                    -3.2223299715802804,
                                    0.07391132329604089,
                                ]
                            ],
                            "event002": [
                                [
                                    -1.4089473532732928,
                                    -0.7167980012223953,
                                    0.5012698554140429,
                                ]
                            ],
                            "event003": [
                                [
                                    0.04458267100509339,
                                    -2.509693827374048,
                                    1.0654793985540807,
                                ]
                            ],
                            "event004": [
                                [
                                    -1.6816351360374835,
                                    -1.9499724241903467,
                                    1.0072285561095513,
                                ]
                            ],
                            "event005": [
                                [
                                    3.1244171993439678,
                                    -4.380451961213775,
                                    -0.37825857170905186,
                                ]
                            ],
                            "event006": [0.0, 0.0, 0.0],
                            "mic000": [
                                -0.8744811866103914,
                                -3.5851696785056086,
                                0.8982316097262095,
                            ],
                            "event007": [
                                [
                                    -0.618300826025485,
                                    -3.3916159562613037,
                                    0.7266494362277474,
                                ]
                            ],
                            "event008": [
                                [
                                    2.52527588716348,
                                    -3.18197875102996,
                                    1.0475087649295827,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    307.47252665895246,
                                    74.55308316945248,
                                    3.700687710629241,
                                ]
                            ],
                            "event001": [
                                [
                                    294.19679075905685,
                                    88.80143053636482,
                                    3.5334754377331783,
                                ]
                            ],
                            "event002": [
                                [
                                    206.96458822145027,
                                    72.40624992075072,
                                    1.6583737476211609,
                                ]
                            ],
                            "event003": [
                                [
                                    271.0177059117361,
                                    66.99981627743632,
                                    2.726865796194692,
                                ]
                            ],
                            "event004": [
                                [
                                    229.22587110867548,
                                    68.63622110254275,
                                    2.764922883210509,
                                ]
                            ],
                            "event005": [
                                [
                                    305.4988832364828,
                                    94.02133843617655,
                                    5.393831825996189,
                                ]
                            ],
                            "event006": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    256.2922914695294,
                                    76.3199538771074,
                                    3.7980230375977007,
                                ]
                            ],
                            "event007": [
                                [
                                    259.66828201943576,
                                    78.09770494436374,
                                    3.5232618564903997,
                                ]
                            ],
                            "event008": [
                                [
                                    308.4361517015339,
                                    75.54052355224887,
                                    4.195149781700265,
                                ]
                            ],
                        },
                    }
                ],
                "event007": [
                    {
                        "alias": "event007",
                        "coordinates_absolute": [
                            4.468979500024191,
                            -0.915794287462468,
                            1.1239507807336935,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    2.788401803689853,
                                    0.560675690545505,
                                    0.2590120118828163,
                                ]
                            ],
                            "event001": [
                                [
                                    2.0662563984128886,
                                    0.1692859846810233,
                                    -0.6527381129317065,
                                ]
                            ],
                            "event002": [
                                [
                                    -0.7906465272478078,
                                    2.6748179550389084,
                                    -0.22537958081370446,
                                ]
                            ],
                            "event003": [
                                [
                                    0.6628834970305784,
                                    0.8819221288872559,
                                    0.33882996232633333,
                                ]
                            ],
                            "event004": [
                                [
                                    -1.0633343100119985,
                                    1.441643532070957,
                                    0.28057911988180395,
                                ]
                            ],
                            "event005": [
                                [
                                    3.7427180253694528,
                                    -0.9888360049524714,
                                    -1.1049080079367992,
                                ]
                            ],
                            "event006": [
                                [
                                    0.618300826025485,
                                    3.3916159562613037,
                                    -0.7266494362277474,
                                ]
                            ],
                            "event007": [0.0, 0.0, 0.0],
                            "mic000": [
                                -0.25618036058490645,
                                -0.1935537222443049,
                                0.17158217349846216,
                            ],
                            "event008": [
                                [
                                    3.143576713188965,
                                    0.20963720523134377,
                                    0.3208593287018353,
                                ]
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    11.369100724126422,
                                    84.79663464674033,
                                    2.8559812798912194,
                                ]
                            ],
                            "event001": [
                                [
                                    4.683715745482692,
                                    107.47654673032133,
                                    2.173508751458174,
                                ]
                            ],
                            "event002": [
                                [
                                    106.46709506714974,
                                    94.61967258263581,
                                    2.798315382349764,
                                ]
                            ],
                            "event003": [
                                [
                                    53.070196844217456,
                                    72.92750737360733,
                                    1.154126039662299,
                                ]
                            ],
                            "event004": [
                                [
                                    126.41192025543236,
                                    81.09820022070937,
                                    1.8132127759654588,
                                ]
                            ],
                            "event005": [
                                [
                                    345.20046867522086,
                                    105.92989982351605,
                                    4.025736773327169,
                                ]
                            ],
                            "event006": [
                                [
                                    79.66828201943578,
                                    101.90229505563627,
                                    3.5232618564903997,
                                ]
                            ],
                            "event007": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    217.07239383102925,
                                    61.8802570473299,
                                    0.36404925876383076,
                                ]
                            ],
                            "event008": [
                                [3.815262000368857, 84.1849304490839, 3.166855383236838]
                            ],
                        },
                    }
                ],
                "event008": [
                    {
                        "alias": "event008",
                        "coordinates_absolute": [
                            1.3254027868352258,
                            -1.1254314926938118,
                            0.8030914520318582,
                        ],
                        "coordinates_relative_cartesian": {
                            "event000": [
                                [
                                    -0.35517490949911235,
                                    0.35103848531416126,
                                    -0.061847316819019005,
                                ]
                            ],
                            "event001": [
                                [
                                    -1.0773203147760766,
                                    -0.04035122055032048,
                                    -0.9735974416335418,
                                ]
                            ],
                            "event002": [
                                [
                                    -3.934223240436773,
                                    2.4651807498075646,
                                    -0.5462389095155398,
                                ]
                            ],
                            "event003": [
                                [
                                    -2.480693216158387,
                                    0.6722849236559121,
                                    0.017970633624498022,
                                ]
                            ],
                            "event004": [
                                [
                                    -4.206911023200964,
                                    1.2320063268396133,
                                    -0.04028020882003136,
                                ]
                            ],
                            "event005": [
                                [
                                    0.5991413121804876,
                                    -1.1984732101838151,
                                    -1.4257673366386345,
                                ]
                            ],
                            "event006": [
                                [
                                    -2.52527588716348,
                                    3.18197875102996,
                                    -1.0475087649295827,
                                ]
                            ],
                            "event007": [
                                [
                                    -3.143576713188965,
                                    -0.20963720523134377,
                                    -0.3208593287018353,
                                ]
                            ],
                            "event008": [0.0, 0.0, 0.0],
                            "mic000": [
                                -3.3997570737738716,
                                -0.40319092747564866,
                                -0.14927715520337315,
                            ],
                        },
                        "coordinates_relative_polar": {
                            "event000": [
                                [
                                    135.33558827919305,
                                    97.0600736096059,
                                    0.5031921353787988,
                                ]
                            ],
                            "event001": [
                                [
                                    182.14502072756912,
                                    132.08481901216635,
                                    1.452631839106008,
                                ]
                            ],
                            "event002": [
                                [147.928778799393, 96.71022323981181, 4.674784014377919]
                            ],
                            "event003": [
                                [
                                    164.83666849227376,
                                    89.59939529768373,
                                    2.5702390540457793,
                                ]
                            ],
                            "event004": [
                                [
                                    163.67717473568678,
                                    90.52646718507565,
                                    4.383784032285943,
                                ]
                            ],
                            "event005": [
                                [
                                    296.5614065030084,
                                    136.778513681845,
                                    1.9565584186819676,
                                ]
                            ],
                            "event006": [
                                [
                                    128.4361517015339,
                                    104.45947644775114,
                                    4.195149781700265,
                                ]
                            ],
                            "event007": [
                                [
                                    183.81526200036885,
                                    95.8150695509161,
                                    3.166855383236838,
                                ]
                            ],
                            "event008": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    186.7633482306259,
                                    92.49666514304788,
                                    3.4268345092431534,
                                ]
                            ],
                        },
                    }
                ],
            },
            "microphones": {
                "mic000": {
                    "name": "ambeovr",
                    "micarray_type": "AmbeoVR",
                    "is_spherical": True,
                    "n_capsules": 4,
                    "capsule_names": ["FLU", "FRD", "BLD", "BRU"],
                    "coordinates_absolute": [
                        [4.730952140262493, -0.7164482855647675, 0.9581043715987417],
                        [4.730952140262493, -0.7280328448715588, 0.9466328428717209],
                        [4.719367580955701, -0.7164482855647675, 0.9466328428717209],
                        [4.719367580955701, -0.7280328448715588, 0.9581043715987417],
                    ],
                    "coordinates_polar": [
                        [45.0, 55.0, 0.01],
                        [315.0, 125.0, 0.01],
                        [135.0, 125.0, 0.01],
                        [225.0, 55.0, 0.01],
                    ],
                    "coordinates_center": [
                        4.725159860609097,
                        -0.7222405652181632,
                        0.9523686072352313,
                    ],
                    "coordinates_cartesian": [
                        [
                            0.005792279653395693,
                            0.005792279653395692,
                            0.005735764363510461,
                        ],
                        [
                            0.00579227965339569,
                            -0.005792279653395693,
                            -0.005735764363510461,
                        ],
                        [
                            -0.005792279653395691,
                            0.005792279653395692,
                            -0.005735764363510461,
                        ],
                        [
                            -0.0057922796533956935,
                            -0.005792279653395692,
                            0.005735764363510461,
                        ],
                    ],
                }
            },
            "mesh": {
                "fname": "Oyens",
                "ftype": ".glb",
                "fpath": utils.get_project_root()
                / "tests/test_resources/meshes/Oyens.glb",
                "units": "meters",
                "from_gltf_primitive": False,
                "name": "defaultobject",
                "node": "defaultobject",
                "bounds": [
                    [-3.0433080196380615, -10.448445320129395, -1.1850370168685913],
                    [5.973234176635742, 2.101027011871338, 2.4577369689941406],
                ],
                "centroid": [1.527919030159762, -4.550817438070386, 1.162934397641578],
            },
            "rlr_config": {
                "diffraction": 1,
                "direct": 1,
                "direct_ray_count": 500,
                "direct_sh_order": 3,
                "frequency_bands": 4,
                "global_volume": 1.0,
                "hrtf_back": [0.0, 0.0, 1.0],
                "hrtf_right": [1.0, 0.0, 0.0],
                "hrtf_up": [0.0, 1.0, 0.0],
                "indirect": 1,
                "indirect_ray_count": 5000,
                "indirect_ray_depth": 200,
                "indirect_sh_order": 1,
                "max_diffraction_order": 10,
                "max_ir_length": 4.0,
                "mesh_simplification": 0,
                "sample_rate": 44100.0,
                "size": 146,
                "source_ray_count": 200,
                "source_ray_depth": 10,
                "temporal_coherence": 0,
                "thread_count": 1,
                "transmission": 1,
                "unit_scale": 1.0,
            },
            "empty_space_around_mic": 0.1,
            "empty_space_around_emitter": 0.2,
            "empty_space_around_surface": 0.2,
            "empty_space_around_capsule": 0.05,
            "repair_threshold": None,
        }
    ],
)
def test_worldstate_from_dict(input_dict: dict):
    wstate = WorldState.from_dict(input_dict)
    assert isinstance(wstate, WorldState)
    # Should have the correct number of emitters and microphones
    assert (
        wstate.ctx.get_source_count()
        == len(wstate.emitters)
        == len(input_dict["emitters"])
    )
    assert wstate.ctx.get_listener_count() == sum(
        ws.n_capsules for ws in wstate.microphones.values()
    )


def test_magic_methods(oyens_space):
    for method in ["__len__", "__str__", "__getitem__", "__repr__"]:
        assert hasattr(oyens_space, method)
        _ = getattr(oyens_space, method)
    # Compare equality
    assert oyens_space == WorldState.from_dict(oyens_space.to_dict())


@pytest.mark.parametrize(
    "starting_position,duration,max_speed,temporal_resolution,raises",
    [
        # Test 1: define a valid starting position, don't define an ending position
        # (np.array([1.5, -4.6, 1.2]), 5.0, 2.0, 1.0, False),
        # Test 2: define an INVALID starting and ending position
        (
            np.array([-1000, 1000, -1000]),
            5.0,
            1.0,
            4,
            True,
        ),
        # Test 3: slow velocity, high duration + resolution
        (None, 10.0, 0.25, 4.0, False),
        # Test 4: high velocity, small duration + resolution
        (None, 0.5, 2.0, 1.0, False),
        # Test 5: high resolution, small duration + velocity
        (None, 1.0, 0.25, 4.0, False),
        # Test 6: small resolution, high duration + velocity
        (None, 10.0, 2.0, 1.0, False),
    ],
)
@pytest.mark.parametrize(
    "shape",
    ["linear", "circular", "random"],
)
def test_define_trajectory(
    starting_position,
    duration,
    max_speed,
    temporal_resolution,
    raises,
    shape,
    oyens_space,
):
    if not raises:
        trajectory = oyens_space.define_trajectory(
            duration=duration,
            starting_position=starting_position,
            shape=shape,
            velocity=max_speed,
            resolution=temporal_resolution,
        )
        assert isinstance(trajectory, np.ndarray)
        assert oyens_space._validate_position(trajectory)

        # Check the shape: expecting (n_points, xyz == 3)
        n_points_actual, n_coords = trajectory.shape
        assert n_coords == 3
        assert n_points_actual >= 2

        # If we've explicitly provided a starting and ending position, these should be maintained in the trajectory
        if starting_position is not None:
            assert np.allclose(trajectory[0, :], starting_position, atol=1e-4)

        # Check that speed constraints are never violated between points
        deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        max_segment_distance = max_speed / temporal_resolution
        assert np.all(deltas <= max_segment_distance + 1e-5)

        # If the shape is linear, check that the distance between all points is roughly equivalent
        if shape == "linear":
            assert np.allclose(deltas, deltas[0], atol=1e-4)

        # Check distance between starting and ending point
        total_distance = np.linalg.norm(trajectory[-1, :] - trajectory[0, :])
        assert total_distance <= (max_speed * duration)

    else:
        with pytest.raises(ValueError):
            _ = oyens_space.define_trajectory(
                duration=duration,
                starting_position=starting_position,
                shape=shape,
                velocity=max_speed,
                resolution=temporal_resolution,
            )


@pytest.mark.parametrize(
    "ref,r,n,raises",
    [
        (np.array([4.73, -0.72, 0.96]), 0.5, 100, False),
        ([1.6, -5.1, 1.7], 1.0, 100, False),
        ([1000, 1000, 1000], 100, 10, True),
    ],
)
def test_get_valid_position_with_max_distance(ref, r, n, raises, oyens_space):
    if not raises:
        point = oyens_space.get_valid_position_with_max_distance(ref, r, n)
        assert isinstance(point, np.ndarray)
        assert np.linalg.norm(point - ref) <= r
    else:
        with pytest.raises(ValueError):
            _ = oyens_space.get_valid_position_with_max_distance(ref, r, n)
