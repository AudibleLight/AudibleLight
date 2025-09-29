#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/worldstate.py"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from trimesh import Scene, Trimesh

from audiblelight import config, utils
from audiblelight.micarrays import (
    MICARRAY_LIST,
    AmbeoVR,
    MonoCapsule,
    sanitize_microphone_input,
)
from audiblelight.worldstate import Emitter, WorldState, load_mesh
from tests import utils_tests


@pytest.mark.parametrize("mesh_fpath", utils_tests.TEST_MESHES)
def test_load_mesh_from_fpath(mesh_fpath: str):
    loaded = load_mesh(mesh_fpath)
    assert isinstance(loaded, Trimesh)
    assert loaded.metadata["fpath"] == str(
        mesh_fpath
    )  # need both to be a string, or we'll get TypeError
    assert loaded.units == config.MESH_UNITS  # units should be in meters
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
    oyens_space.add_emitter(position, emitter_alias, mic=None, keep_existing=False)
    assert isinstance(oyens_space.emitters, dict)
    assert oyens_space.num_emitters == 1
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
        # "coordinates_relative_cartesian",
        # "coordinates_relative_polar",
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
            mic="ambeovr", position=[1000, 1000, 1000], keep_existing=False
        )
    # Raise error when trying to add emitter out of bounds
    oyens_space.add_microphone(alias="ambeovr")
    with pytest.raises(ValueError):
        oyens_space.add_emitter(
            mic="ambeovr", position=[1000, 1000, 1000], keep_existing=False
        )
    # Cannot add emitter that directly intersects with a microphone
    oyens_space.add_microphone(position=[-0.5, -0.5, 0.5], keep_existing=False)
    with pytest.raises(ValueError):
        oyens_space.add_emitter([-0.5, -0.5, 0.5])  # same, in absolute terms
    with pytest.raises(ValueError):
        oyens_space.add_emitter(
            [0.0, 0.0, 0.0], mic="mic000"
        )  # same, in relative terms
    # This emitter is valid, but has no direct path to the microphone
    with pytest.raises(ValueError):
        # emitter is in bedroom 2, microphone is in living room
        oyens_space.add_microphone(
            position=np.array([-1.5, -1.5, 0.7]), alias="tester", keep_existing=False
        )
        oyens_space.add_emitter(
            position=np.array([2.9, -7.0, 0.3]),
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
                position=position, mic="tester", keep_existing=False
            )
    else:
        oyens_space.add_emitter(position=position, mic="tester", keep_existing=False)
        assert oyens_space.num_emitters == 1
        src = oyens_space.get_emitter("src000", 0)
        assert isinstance(src, Emitter)
        # coordinates_relative dict should be as expected
        assert np.allclose(
            src.coordinates_relative_cartesian["tester"], position, atol=utils.SMALL
        )
        assert np.allclose(
            src.coordinates_relative_polar["tester"],
            utils.cartesian_to_polar(position),
            atol=utils.SMALL,
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
    oyens_space.add_emitters(positions, emitter_aliases, keep_existing=False)
    assert oyens_space.num_emitters == len(positions)
    if emitter_aliases is not None:
        assert set(oyens_space.emitters.keys()) == set(emitter_aliases)
    for emitter_list in oyens_space.emitters.values():
        for emitter in emitter_list:
            assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)


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
        aliases=emit_aliases,
    )
    assert oyens_space.num_emitters == sum(expected)
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
                    atol=utils.SMALL,
                )
                assert np.allclose(
                    emitter.coordinates_relative_polar["testmic"],
                    utils.cartesian_to_polar(position),
                    atol=utils.SMALL,
                )
                assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)


@pytest.mark.parametrize(
    "n_emitters",
    [i for i in range(1, 10, 2)],
)
def test_add_n_emitters(n_emitters, oyens_space: WorldState):
    oyens_space.add_emitters(n_emitters=n_emitters, keep_existing=False)
    assert oyens_space.num_emitters == n_emitters
    for emitter_list in oyens_space.emitters.values():
        for emitter in emitter_list:
            assert oyens_space._is_point_inside_mesh(emitter.coordinates_absolute)
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
        aliases=emit_alias,
    )
    assert oyens_space.num_emitters == sum(expected)
    for position, is_added, alias in zip(test_position, expected, emit_alias):
        if is_added:
            for emitter in oyens_space[alias]:
                assert np.allclose(
                    emitter.coordinates_absolute, position, atol=utils.SMALL
                )


def test_add_emitters_invalid(oyens_space: WorldState):
    # n_emitters must be a postive integer
    with pytest.raises(AssertionError):
        for inp in ["asdf", [], -1, -100, 0]:
            oyens_space.add_emitters(
                n_emitters=inp,
                keep_existing=False,
            )
    # Cannot specify both a number of random emitters and positions for them
    with pytest.raises(TypeError):
        oyens_space.add_emitters(positions=[[0, 0, 0]], n_emitters=1)
    # Cannot add emitters that are way outside the mesh
    with pytest.raises(ValueError):
        oyens_space.add_emitters(
            [[1000.0, 1000.0, 1000.0], [-1000, -1000, -1000]],
            keep_existing=False,
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
    # Try with a point outside the mesh: should be NaN
    bad_point = np.array([1000, 1000, 1000])
    result = oyens_space.calculate_weighted_average_ray_length(
        bad_point, num_rays=num_rays
    )
    assert np.isnan(result)


@pytest.mark.parametrize("test_num", range(1, 5))
def test_get_random_position(test_num: int, oyens_space: WorldState):
    # For reproducible results
    utils.seed_everything(test_num)
    # Add some microphones and emitters to the space
    for idx_ in range(test_num):
        oyens_space.add_microphone(keep_existing=True)
        oyens_space.add_emitter(keep_existing=True)
    # Grab a random position
    random_point = oyens_space.get_random_position()
    # It should be valid (suitable distance from surfaces, inside mesh, away from mics/emitters...)
    assert oyens_space._validate_position(random_point)
    assert random_point.shape == (3,)  # should be a 1D array of XYZ


def test_get_random_position_with_weighted_average_ray_length(oyens_space: WorldState):
    # Test that, when this parameter is true, any random position has a minimum ray length
    oyens_space.ensure_minimum_weighted_average_ray_length = True
    oyens_space.minimum_weighted_average_ray_length = 1.0
    point = oyens_space.get_random_position()
    ray_length = oyens_space.calculate_weighted_average_ray_length(point, num_rays=1000)
    assert ray_length >= oyens_space.minimum_weighted_average_ray_length
    # Reset back to default
    oyens_space.ensure_minimum_weighted_average_ray_length = False


def test_create_plot(oyens_space):
    # Add some microphones and emitters
    oyens_space.add_microphone(keep_existing=False)
    oyens_space.add_emitter()
    # Create the plot
    fig = oyens_space.create_plot()
    assert isinstance(fig, plt.Figure)
    # Should have two axes for the two views
    assert len(fig.get_axes()) == 2


def test_create_scene(oyens_space):
    # Add some microphones and emitters
    oyens_space.add_microphone(keep_existing=False)
    oyens_space.add_emitter()
    # Create the scene
    scene = oyens_space.create_scene()
    assert isinstance(scene, Scene)
    # Should have more geometry than the "raw" scene (without adding spheres for capsules/emitters)
    assert len(scene.geometry) > len(oyens_space.mesh.scene().geometry)


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
    "cfg,expected",
    [
        (dict(sample_rate=22050, global_volume=0.5), None),
        (dict(will_raise="an_error", sample_rate=595959), AttributeError),
    ],
)
def test_config_parse(cfg, expected):
    if expected is None:
        space = WorldState(mesh=str(utils_tests.TEST_MESHES[-1]), rlr_kwargs=cfg)
        for ke, val in cfg.items():
            assert getattr(space.ctx.config, ke) == val

    else:
        with pytest.raises(expected):
            _ = WorldState(mesh=str(utils_tests.TEST_MESHES[-1]), rlr_kwargs=cfg)


def test_to_dict(oyens_space: WorldState):
    oyens_space.add_microphone(
        microphone_type="ambeovr", alias="tester_mic", keep_existing=False
    )
    oyens_space.add_emitter(alias="tester_emitter", keep_existing=False)
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
        (np.array([-90.0, 0.0, 0.2]), True, True, True),
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
        assert np.allclose(placed_at, position, atol=utils.SMALL)
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
                "tester_emitter": [
                    [
                        0.8212208994051426,
                        -6.006953079901214,
                        0.5535639680338247,
                    ],
                ]
            },
            "microphones": {
                "tester_mic": {
                    "name": "ambeovr",
                    "micarray_type": "AmbeoVR",
                    "is_spherical": True,
                    "n_capsules": 4,
                    "channel_layout_type": "mono",
                    "capsule_names": ["FLU", "FRD", "BLD", "BRU"],
                    "coordinates_absolute": [
                        [-0.38804551239949914, -8.630788873071257, 1.4665762923251686],
                        [-0.38804551239949914, -8.642373432378047, 1.4551047635981476],
                        [-0.3996300717062905, -8.630788873071257, 1.4551047635981476],
                        [-0.3996300717062905, -8.642373432378047, 1.4665762923251686],
                    ],
                    "coordinates_center": [
                        -0.3938377920528948,
                        -8.636581152724652,
                        1.460840527961658,
                    ],
                }
            },
            "mesh": {
                "fname": "Oyens",
                "ftype": ".glb",
                "fpath": str(utils_tests.OYENS_PATH),
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
        == wstate.num_emitters
        == len(input_dict["emitters"])
    )
    assert wstate.ctx.get_listener_count() == sum(
        ws.n_capsules for ws in wstate.microphones.values()
    )
    assert hasattr(
        wstate.get_emitter("tester_emitter", 0), "coordinates_relative_polar"
    )


def test_worldstate_magic_methods(oyens_space):
    for method in ["__len__", "__str__", "__getitem__", "__repr__"]:
        assert hasattr(oyens_space, method)
        _ = getattr(oyens_space, method)
    # Compare equality
    assert oyens_space == WorldState.from_dict(oyens_space.to_dict())


def test_emitter_magic_methods(oyens_space):
    em = Emitter(
        alias="asdf", coordinates_absolute=oyens_space.get_random_point_inside_mesh()
    )
    for method in ["__len__", "__str__", "__getitem__", "__repr__"]:
        assert hasattr(oyens_space, method)
        _ = getattr(oyens_space, method)
    # Compare equality
    outdict = em.to_dict()
    assert em == Emitter.from_dict(outdict)
    # Remove a key, should raise an error
    with pytest.raises(KeyError):
        outdict.pop("alias")
        _ = Emitter.from_dict(outdict)


@pytest.mark.parametrize(
    "starting_position,duration,max_speed,temporal_resolution,raises",
    [
        # Test 1: define an INVALID starting
        (
            np.array([-1000, 1000, -1000]),
            5.0,
            1.0,
            4,
            "Invalid starting position",
        ),
        # Test 2: slow velocity, high duration + resolution
        (None, 10.0, 0.25, 4.0, False),
        # Test 3: high velocity, small duration + resolution
        (None, 0.5, 2.0, 1.0, False),
        # Test 4: high resolution, small duration + velocity
        (None, 1.0, 0.25, 4.0, False),
        # Test 5: small resolution, high duration + velocity
        (None, 5.0, 2.0, 1.0, False),
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
    oyens_space.clear_emitters()

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
            assert np.allclose(trajectory[0, :], starting_position, atol=utils.SMALL)

        # Check that speed constraints are never violated between points
        deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        max_segment_distance = max_speed / temporal_resolution
        assert np.all(deltas <= max_segment_distance + 1e-5)

        # If the shape is linear, check that the distance between all points is roughly equivalent
        if shape == "linear":
            assert np.allclose(deltas, deltas[0], atol=utils.SMALL)

        # Check distance between starting and ending point
        total_distance = np.linalg.norm(trajectory[-1, :] - trajectory[0, :])
        assert total_distance <= (max_speed * duration)

        # If we add the emitters to the state, check that we have the correct number
        oyens_space._add_emitters_without_validating(trajectory, alias="tmp")
        assert len(oyens_space.get_emitters("tmp")) == len(trajectory)

    else:
        with pytest.raises(ValueError, match=raises):
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


def test_add_foa_capsule(oyens_space):
    # Add many different types of microphone in
    oyens_space.add_microphone(
        microphone_type="foalistener",
        position=[-0.5, -0.5, 0.5],
        keep_existing=False,
        alias="foa_tester",
    )
    oyens_space.add_microphone(
        microphone_type="monocapsule", keep_existing=True, alias="mono_tester"
    )
    oyens_space.add_microphone(
        microphone_type="ambeovr", keep_existing=True, alias="ambeo_tester"
    )

    # Add two emitters in
    oyens_space.add_emitter(alias="tester")
    oyens_space.add_emitter(keep_existing=True, alias="tester2")

    # Simulate the IR
    oyens_space.simulate()

    # Grab FOA microphone
    mic = oyens_space.get_microphone("foa_tester")
    assert mic.channel_layout_type == "foa"

    # Test all microphones IRs as expected
    for mic in oyens_space.microphones.values():
        n_caps, n_emits, n_samps = mic.irs.shape
        assert n_caps == mic.n_capsules
        assert n_emits == 2
        assert n_samps >= 1
        # Should not be just zeroes
        assert not np.all(mic.irs == 0)


@pytest.mark.parametrize("n_emitters", range(2, 4))
@pytest.mark.parametrize("normalize", [True, False])
def test_ir_normalization(n_emitters, normalize, oyens_space):
    # Add microphone and emitters
    oyens_space.add_microphone(
        microphone_type="ambeovr", keep_existing=True, alias="ambeo_tester"
    )
    for _ in range(n_emitters):
        oyens_space.add_emitter(keep_existing=True)

    # Simulate and grab IRs
    oyens_space.simulate(normalize=normalize)
    ir_out = oyens_space.irs["ambeo_tester"]

    # Relative energies of all IRs should be centered around 1 if normalizing
    #  Transposing just means that we get the IRs for each event individually
    for event_irs in ir_out.transpose(1, 0, 2):
        energies = np.sqrt(np.sum(np.power(np.abs(event_irs), 2), axis=-1))
        mean_energy = np.mean(energies)
        if normalize:
            assert pytest.approx(mean_energy) == 1.0
        else:
            assert not pytest.approx(mean_energy) == 1
