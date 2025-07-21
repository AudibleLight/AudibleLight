#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/worldstate.py"""

import json
import os
from tempfile import TemporaryDirectory

import pytest
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from trimesh import Trimesh, Scene
from scipy.signal import stft
from pyroomacoustics.doa.music import MUSIC

from audiblelight import utils
from audiblelight.worldstate import WorldState, load_mesh, repair_mesh, Emitter
from audiblelight.micarrays import MICARRAY_LIST, AmbeoVR, MonoCapsule, sanitize_microphone_input


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
    assert loaded.metadata["fpath"] == str(mesh_fpath)    # need both to be a string, or we'll get TypeError
    assert loaded.units == utils.MESH_UNITS    # units should be in meters
    # If we try to load from a mesh object, should raise an error
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = load_mesh(loaded)


@pytest.mark.parametrize("mesh_fpath,expected", [("iamnotafile", FileNotFoundError), (1234, TypeError)])
def test_load_broken_mesh(mesh_fpath: str, expected):
    with pytest.raises(expected):
        load_mesh(mesh_fpath)


@pytest.mark.parametrize(
    "microphone_type,position,alias",
    [
        (None, None, None),    # places 1 mono mic in a random position with default alias
        (None, [-0.5, -0.5, 0.5], None),  # places mono mic in assigned position
        ("eigenmike32", None, "customalias000"),     # places eigenmike32 in random position
        (AmbeoVR, None, "ambeovr000"),  # same but with AmbeoVR
        ("eigenmike32", [-0.5, -0.5, 0.5], "customeigenmike000"),    # places eigenmike32 in assigned position
        ("ambeovr", [-0.1, -0.1, 0.6], "ambeovr000"),    # places AmbeoVR in assigned position
    ]
)
def test_add_microphone(microphone_type, position, alias, oyens_space: WorldState):
    """Test adding a single microphone to the space"""
    # Add the microphones to the space: keep_existing=False ensures we remove previously-added microphones
    oyens_space.add_microphone(microphone_type, position, alias, keep_existing=False)
    assert isinstance(oyens_space.microphones, dict)
    assert len(oyens_space.microphones) == 1    # only addding one microphone with this function
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
        assert list(oyens_space.microphones.keys())[0] == alias     # alias should be what we expect
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
    for inp in [[1000., 1000., 1000.], [-1000, -1000, -1000]]:
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
        (["eigenmike32", "eigenmike32"], np.array([[-0.5, -0.5, 0.5], [0.6, 0.4, 0.4] ]), None),
        # Eigenmike32 and ambeovr in assigned positions with assigned aliases
        (["eigenmike32", "ambeovr"], [[-0.5, -0.5, 0.5], [-0.1, -0.1, 0.6]], ["eigenmike000", "ambeovr000"]),
        # Three ambeovrs in random positions with default aliases
        (["ambeovr", "ambeovr", AmbeoVR], None, None),
        # A mono capsule, an ambeovr, and an eigenmike (all specified in different ways) with random positions
        ([None, AmbeoVR, "eigenmike32"], None, ["mono", "ambeo", "eigen"])
    ]
)
def test_add_microphones(microphone_types, positions, aliases, oyens_space: WorldState):
    # Add microphone types into space
    oyens_space.add_microphones(microphone_types, positions, aliases, keep_existing=False, raise_on_error=True)
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
        for expected_pos, actual_mic in zip(positions, oyens_space.microphones.values()):
            assert np.array_equal(expected_pos, actual_mic.coordinates_center)


# noinspection PyTypeChecker
def test_add_microphones_invalid_inputs(oyens_space: WorldState):
    # Trying to add non-unique aliases raises an error
    with pytest.raises(ValueError):
        oyens_space.add_microphones(None, None, ["ambeovr", "ambeovr"], keep_existing=False)
    # Trying to add iterables with different lengths raises an error
    with pytest.raises(ValueError):
        oyens_space.add_microphones(["ambeovr", "ambeovr"], None, ["ambeovr"], keep_existing=False)
    # Trying to add microphone outside the mesh
    for pos in [[-1000, -1000, -1000], [1000, 1000, 1000]]:
        with pytest.raises(ValueError):
            oyens_space.add_microphones(["ambeovr"], [pos], ["ambeovr"], keep_existing=False, raise_on_error=True)
    # Cannot add alias that is already in the dictionary
    oyens_space.add_microphones(aliases=["tmp_alias"])
    with pytest.raises(KeyError):
        oyens_space.add_microphones(aliases=["tmp_alias"], keep_existing=True)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "test_position,expected",
    [
        (np.array([-0.4, -0.5, 0.5]), False),    # Too close to mic
        (np.array([-0.5, -0.4, 0.5]), False),    # Too close to mic
        (np.array([-0.5, -0.5, 0.4]), False),    # Too close to mic
        (np.array([-0.8, -1.5, 0.2]), False),    # Too close to the surface
        (np.array([-0.1, -0.1, 0.6]), True),    # Fine!
        (np.array([0.5, 0.5, 0.5]), True),   # Also fine
        (np.array([0.5]), ValueError),     # should raise an error with invalid array shape
        (np.array([[0.5, 0.5, 0.5], [-0.4, -0.5, 0.5]]), False),     # 1 invalid, 2 valid
        (np.array([[0.5, 0.5, 0.5], [-0.1, -0.1, 0.6]]), True),    # both valid
        (np.array([[0.5], [0.5]]), ValueError),     # should raise an error with invalid array shape
    ]
)
def test_validate_position(test_position: np.ndarray, expected: bool, oyens_space: WorldState):
    """Given a microphone with coordinates [-0.5, -0.5, 0.5], test whether test_position is valid"""
    oyens_space.add_microphone(microphone_type="ambeovr", position=[-0.5, -0.5, 0.5], keep_existing=False)
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            oyens_space._validate_position(test_position)
    else:
        assert oyens_space._validate_position(test_position) == expected


@pytest.mark.parametrize(
    "position,emitter_alias",
    [
        (None, None),    # Add random emitter with no aliases
        ([-0.1, -0.1, 0.6], "custom_alias"),    # add specific emitter with custom alias
        (np.array([-0.5, -0.5, 0.5]), "custom_alias"),    # position as array, not list
    ]
)
def test_add_emitter(position, emitter_alias, oyens_space: WorldState):
    oyens_space._clear_microphones()
    # Add the emitters in and check that the shape of the resulting array is what we expect
    oyens_space.add_emitter(position, emitter_alias, mic=None, keep_existing=False, polar=False)
    assert isinstance(oyens_space.emitters, dict)
    assert len(oyens_space.emitters) == 1
    # Get the desired emitter: should be the first element in the list
    src = oyens_space.get_emitter(emitter_alias if emitter_alias is not None else "src000", 0)
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
    for k in ["alias", "coordinates_absolute", "coordinates_relative_cartesian", "coordinates_relative_polar"]:
        assert k in di.keys()
    # Test output strings
    assert isinstance(repr(src), str)
    assert isinstance(str(src), str)
    assert repr(src) != str(src)


def test_add_emitter_invalid(oyens_space: WorldState):
    # Raise error when no microphone with alias has been added
    with pytest.raises(KeyError):
        oyens_space.add_emitter(mic="ambeovr", position=[1000, 1000, 1000], keep_existing=False, polar=False)
    # Raise error when trying to add emitter out of bounds
    oyens_space.add_microphone(alias="ambeovr")
    with pytest.raises(ValueError):
        oyens_space.add_emitter(mic="ambeovr", position=[1000, 1000, 1000], keep_existing=False, polar=False)
    # Cannot add emitter that directly intersects with a microphone
    oyens_space.add_microphone(position=[-0.5, -0.5, 0.5], keep_existing=False)
    with pytest.raises(ValueError):
        oyens_space.add_emitter([-0.5, -0.5, 0.5], polar=False)    # same, in absolute terms
    with pytest.raises(ValueError):
        oyens_space.add_emitter([0.0, 0.0, 0.0], mic="mic000", polar=False)    # same, in relative terms
    # Must provide a reference microphone when using polar emitters
    with pytest.raises(AssertionError):
        oyens_space.add_emitter([0.0, 0.0, 0.0], polar=True, mic=None)
    # Cannot use random positions with polar = True
    with pytest.raises(AssertionError):
        oyens_space.add_emitter(position=None, polar=True)
    # This emitter is valid, but has no direct path to the microphone
    with pytest.raises(ValueError):
        # emitter is in bedroom 2, microphone is in living room
        oyens_space.add_microphone(position=np.array([-1.5, -1.5, 0.7]), alias="tester", keep_existing=False)
        oyens_space.add_emitter(
            position=np.array([2.9, -7.0, 0.3]),
            polar=False,
            ensure_direct_path="tester",
            keep_existing=False
        )


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "inputs,outputs",
    [
        (True, ["tester1", "tester2", "tester3"]),
        ("tester1", ["tester1"]),
        (["tester2", "tester3"], ["tester2", "tester3"]),
        (["tester3", "tester3"], ["tester3"]),    # duplicates removed
        (False, []),
        ("tester4", KeyError),    # not a microphone alias
        (["tester1", "tester2", "tester4"], KeyError),     # contains a missing alias
        (object, TypeError),    # cannot handle this type
        (123, TypeError),    # cannot handle this type
    ]
)
def test_get_microphones_from_alias(inputs, outputs, oyens_space: WorldState):
    oyens_space.add_microphones(aliases=["tester1", "tester2", "tester3"], keep_existing=False)
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
    ]
)
def test_add_polar_emitter(emitter_position, expected_position, oyens_space: WorldState):
    oyens_space.add_microphone(
        keep_existing=False,
        position=[-0.5, -0.5, 0.5],
        microphone_type="monocapsule",
        alias="tester"
    )
    oyens_space.add_emitter(position=emitter_position, polar=True, mic="tester", keep_existing=False, alias="testsrc")
    assert np.allclose(oyens_space.get_emitter("testsrc", 0).coordinates_absolute, expected_position, atol=1e-4)


@pytest.mark.parametrize(
    "position,accept",
    [
        ([0.1, 0.0, 0.0], False),
        (np.array([0.0, 0.1, 0.0]), False),
        ([1000, 1000, 1000], False),
        ([-0.2, 0.2, 0.2], True),
        ([0.2, -0.3, -0.2], True),
    ]
)
def test_add_emitter_relative_to_mic(position, accept: bool, oyens_space: WorldState):
    # Add a microphone to the space
    oyens_space.add_microphone(
        microphone_type="ambeovr",
        position=[-0.5, -0.5, 0.5],
        alias="tester",
        keep_existing=False
    )
    # Trying to add an emitter that should be rejected
    if not accept:
        with pytest.raises(ValueError):
            oyens_space.add_emitter(position=position, mic="tester", keep_existing=False, polar=False)
    else:
        oyens_space.add_emitter(position=position, mic="tester", keep_existing=False, polar=False)
        assert len(oyens_space.emitters) == 1
        src = oyens_space.get_emitter("src000", 0)
        assert isinstance(src, Emitter)
        # coordinates_relative dict should be as expected
        assert np.allclose(src.coordinates_relative_cartesian["tester"], position, atol=1e-4)
        assert np.allclose(
            src.coordinates_relative_polar["tester"],
            utils.cartesian_to_polar(position),
            atol=1e-4
        )


@pytest.mark.parametrize(
    "positions,emitter_aliases",
    [
        (np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]), None),
        (np.array([[0.5, 0.5, 0.5], [0.6, 0.2, 0.5]]), ["custom_alias1", "custom_alias2"]),
        ([[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5], [-0.4, -0.5, 0.5]], ["custom_alias1", "custom_alias2", "custom_alias3"]),
    ]
)
def test_add_emitters(positions, emitter_aliases, oyens_space: WorldState):
    oyens_space._clear_microphones()
    oyens_space.add_emitters(positions, emitter_aliases, keep_existing=False, polar=False)
    assert len(oyens_space.emitters) == len(positions)
    if emitter_aliases is not None:
        assert set(oyens_space.emitters.keys()) == set(emitter_aliases)
        # Should have all the other emitters in our relative coords dict
        for emitter_list in oyens_space.emitters.values():
            for emitter in emitter_list:
                assert set(emitter.coordinates_relative_cartesian.keys()) == set(emitter_aliases)
                assert set(emitter.coordinates_relative_polar.keys()) == set(emitter_aliases)
    for emitter_list in oyens_space.emitters.values():
        for emitter in emitter_list:
            assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)



@pytest.mark.parametrize(
    "emitter_positions,expected_positions",
    [
        # 1. Azimuth = 0°, Colatitude = 90° (x+), and Colatitude = 0° (z+)
        # emitter 1: offset 20 cm along +x; emitter 2: offset 20 cm directly above mic
        ([[0.0, 90.0, 0.2], [0.0, 0.0, 0.2]],
         [[-0.3, -0.5, 0.5], [-0.5, -0.5, 0.7]]),
        # 2. Azimuth = 90°, Colatitude = 90° (y+), and Azimuth = 270°, Colatitude = 90° (y−)
        # emitter 1: offset 20 cm along +y; emitter 2: offset 20 cm along −y
        ([[90.0, 90.0, 0.2], [270.0, 90.0, 0.2]],
         [[-0.5, -0.3, 0.5], [-0.5, -0.7, 0.5]]),
    ]
)
def test_add_polar_emitters(emitter_positions, expected_positions, oyens_space: WorldState):
    oyens_space.add_microphone(
        keep_existing=False,
        position=[-0.5, -0.5, 0.5],
        microphone_type="monocapsule",
        alias="tester"
    )
    oyens_space.add_emitters(positions=emitter_positions, polar=True, mics="tester", keep_existing=False)
    for emitter_list, expected_position in zip(oyens_space.emitters.values(), expected_positions):
        for emitter in emitter_list:
            assert np.allclose(emitter.coordinates_absolute, expected_position, atol=1e-4)


@pytest.mark.parametrize(
    "test_position,expected",
    [
        (np.array([[0.1, 0.0, 0.0], [-0.2, 0.2, 0.2]]), (False, True)),    # 1: too close to mic, so skipped, 2: fine
        ([[-0.2, 0.2, 0.2], [-0.2, 0.3, 0.2]], (True, False)),    # 1: fine, 2: too close to emitter 1, so skipped
        (np.array([[-0.2, 0.2, 0.2], [0.2, -0.3, -0.2]]), (True, True)),  # both fine
    ]
)
def test_add_emitters_relative_to_mic(test_position: np.ndarray, expected: tuple[bool], oyens_space: WorldState):
    # Clear everything out
    oyens_space._clear_microphones()
    oyens_space._clear_emitters()
    oyens_space.add_microphone(microphone_type=AmbeoVR, position=[-0.5, -0.5, 0.5], alias="testmic", keep_existing=False)
    # Add the emitters in and check that the shape of the resulting array is what we expect
    #  We set `raise_on_error=False` so we skip over raising an error for invalid emitters
    emit_aliases = [f"test{i}" for i in range(len(test_position))]
    oyens_space.add_emitters(
        positions=test_position,
        mics="testmic",
        keep_existing=False,
        raise_on_error=False,
        polar=False,
        aliases=emit_aliases
    )
    assert len(oyens_space.emitters) == sum(expected)
    for position, is_added, alias in zip(test_position, expected, emit_aliases):
        if is_added:
            emitter_list = oyens_space[alias]    # can also get emitters in this way, too :)
            # Relative position dictionary should be as we expect
            for emitter in emitter_list:
                assert np.allclose(
                    emitter.coordinates_relative_cartesian["testmic"],
                    position,
                    atol=1e-4
                )
                assert np.allclose(
                    emitter.coordinates_relative_polar["testmic"],
                    utils.cartesian_to_polar(position),
                    atol=1e-4
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
        (np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]), (False, True)),    # 1: too close to mic, 2: fine
        (np.array([[0.5, 0.5, 0.5], [0.6, 0.4, 0.5]]), (True, False)),    # 1: fine, 2: too close to emitter 1
        ([[-0.1, -0.1, 0.6]], (True,)),
        ([[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5]], (True, True)),
    ]
)
def test_add_emitters_at_specific_position(test_position: np.ndarray, expected: tuple[bool], oyens_space: WorldState):
    oyens_space.add_microphone(microphone_type=AmbeoVR, position=[-0.5, -0.5, 0.5], keep_existing=False)
    # Add the emitters in and check that the shape of the resulting array is what we expect
    emit_alias = [f"emit{i}" for i in range(len(test_position))]
    oyens_space.add_emitters(
        positions=test_position,
        keep_existing=False,
        raise_on_error=False,
        polar=False,
        aliases=emit_alias
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
    with pytest.raises(ValueError):
        oyens_space.add_emitters(aliases=["asdf", "asdf"], polar=False)
    # Cannot add emitters that are way outside the mesh
    with pytest.raises(ValueError):
        oyens_space.add_emitters([[1000., 1000., 1000.], [-1000, -1000, -1000]], keep_existing=False, polar=False)


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
    assert random_point.shape == (3,)   # should be a 1D array of XYZ


# Goes (1 mic, 4 emitters), (2 mics, 3 emitters), (3 mics, 2 emitters), (4 mics, 1 emitter)
@pytest.mark.parametrize("n_mics,n_emitters", [(m, s) for m, s in zip(list(range(1, 5))[::-1], range(1, 5))])
def test_simulated_ir(n_mics: int, n_emitters: int, oyens_space: WorldState):
    # For reproducible results
    utils.seed_everything(n_emitters)
    # Add some emitters and microphones to the space
    #  We could use other microphone types, but they're slow to simulate
    oyens_space.add_microphones(microphone_types=["ambeovr" for _ in range(n_mics)], keep_existing=False)
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
        assert actual_samples >= 1    # difficult to test number of samples
        total_capsules += actual_capsules
    # IRs for all microphones should have same number of emitters and samples
    _, mic_1_emitters, mic_1_samples = oyens_space.get_microphone("mic000").irs.shape
    assert all([m.irs.shape[1] == mic_1_emitters for m in oyens_space.microphones.values()])
    assert all([m.irs.shape[2] == mic_1_samples for m in oyens_space.microphones.values()])
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


def test_save_wavs(oyens_space: WorldState):
    # Add some microphones and emitters
    oyens_space.add_microphone(microphone_type="ambeovr", keep_existing=False)    # just adds an ambeovr mic in a random plcae
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
            y, _ = sf.read(fp, )
            # Compare to the original IR
            x = oyens_space.irs["mic000"][caps_idx][0]
            assert np.allclose(y, x, atol=1e-4)
    # Temporary directory is implicitly cleaned up


@pytest.mark.parametrize(
    "point_a,point_b,expected_result",
    [
        # Point A in bedroom 1, point B in bedroom 2: should have no direct line
        (
            np.array([-1.5, -1.5, 0.7]),
            np.array([2.9, -7.0, 0.3]),
            False
        ),
        # Point A and B both in living room, should have a direct line
        (
            [2.5, 0., 0.5],
            [2.4, -1.0, 0.7],
            True
        ),
        # Point A in living room, point B in bedroom 2
        (
            np.array([2.5, 0., 0.5]),
            [2.9, -7.0, 0.3],
            False
        )
    ]
)
def test_path_between_points(point_a: np.ndarray, point_b: np.ndarray, expected_result: bool, oyens_space: WorldState):
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
            [-1.5, -1.5, 0.7],    # mic placed in bedroom 1
            [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
            [90, 270]
        ),
        # Test case 2: two emitters at 0 and 180 degree angles from the mic
        (
            [2.9, -7.0, 0.3],    # mic placed in bedroom 2
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [0, 180]
        ),
        # Test case 3: single sound emitter at a 45-degree angle
        (
            [2.5, -1.0, 0.5],  # mic placed in living room
            [[1.0, 1.0, 0.0]],
            [45,]
        )
    ]
)
def test_simulated_doa_with_music(microphone: list, emitters: list, actual_doa: list[int], oyens_space: WorldState):
    """
    Tests DOA of simulated sound emitters and microphones with MUSIC algorithm.

    Places an Eigenmike32, simulates sound emitters, runs MUSIC, checks that estimated DOA is near to actual DOA
    """
    # Add the microphones and simulate the space
    oyens_space.add_microphone(microphone_type="eigenmike32", position=microphone, keep_existing=False, alias="tester")
    oyens_space.add_emitters(positions=emitters, mics="tester", keep_existing=False, polar=False)
    oyens_space.simulate()
    # TODO: in the future we should use simulated sound emitters, not the IRs
    output = oyens_space.irs

    # Create the MUSIC object
    L = oyens_space.get_microphone("tester").coordinates_absolute.T    # coordinates of our capsules for the eigenmike
    fs = int(oyens_space.ctx.config.sample_rate)
    nfft = 1024
    num_emitters = len(oyens_space.emitters)    # number of sound emitters we've added
    assert num_emitters == len(actual_doa) == len(emitters)    # sanity check everything
    music = MUSIC(
        L=L,
        fs=fs,
        nfft=nfft,
        azimuth=np.deg2rad(np.arange(360)),
        num_sources=num_emitters
    )

    # Iterating over all of our sound emitters
    for doa_deg_true, emitter_idx in zip(actual_doa, range(num_emitters)):
        # Get the IRs for this emitter: shape (N_capsules=32, 1=mono, N_samples)
        signals = np.vstack([m[:, emitter_idx, :] for m in output.values()])
        # Iterate over each individual IR (one per capsule: shape = 1, N_samples) and compute the STFT
        #  Stacked shape is (N_capsules, (N_fft / 2) + 1, N_frames)
        stft_signals = np.stack([stft(cs, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2] for cs in signals])
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
    ]
)
def test_simulated_sound_distance(closemic_position: list, farmic_position: list, emitter_position: list, oyens_space):
    """
    Tests distance of simulated sound emitters and microphones.

    Places a emitter and two AmbeoVR microphones near and far, then checks that the sound hits the close mic before far
    """

    oyens_space._clear_microphones()
    oyens_space._clear_emitters()
    # Add the microphones and simulate the space
    oyens_space.add_microphones(
        microphone_types=["ambeovr", AmbeoVR],
        positions=[closemic_position, farmic_position],
        aliases=["closemic", "farmic"],
        keep_existing=False
    )
    oyens_space.add_emitter(emitter_position, mic="closemic", keep_existing=False, polar=False)
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
        (dict(will_raise="an_error", sample_rate=595959), AttributeError)
    ]
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
    oyens_space.add_microphone(microphone_type="ambeovr", alias="tester_mic", keep_existing=False)
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
