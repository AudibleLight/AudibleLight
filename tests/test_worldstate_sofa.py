#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for SOFA functionality inside audiblelight/worldstate.py"""

from unittest.mock import Mock

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.worldstate import Emitter, WorldState, WorldStateSOFA
from tests import utils_tests


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(position=None, keep_existing=True, alias="tmp"),
        dict(
            position=[2.5, 0.0, 0.0],
            keep_existing=False,
        ),
    ],
)
def test_add_emitter(kwargs, daga_space: WorldStateSOFA):
    # Add the emitter in with desired arguments
    daga_space.add_emitter(**kwargs)

    # Should be a dictionary with one emitter
    assert isinstance(daga_space.emitters, dict)
    assert daga_space.num_emitters == 1 == len(daga_space.emitters)

    # Get the desired emitter: should be the first element in the list
    emitter_alias = kwargs.get("alias", "src000")
    src = daga_space.get_emitter(kwargs.get("alias", "src000"), 0)

    # Should be an emitter object
    assert isinstance(src, Emitter)
    # Should have all the desired attributes
    assert src.alias == emitter_alias

    # Actual position should be within the SOFA file
    actual_pos = src.coordinates_absolute
    assert isinstance(actual_pos, np.ndarray)
    assert actual_pos in daga_space.get_source_positions()

    # If we've provided a position, actual one should be close
    expected_pos = kwargs.get("position", None)
    if expected_pos:
        assert np.allclose(expected_pos, src.coordinates_absolute, atol=utils.SMALL)

    # Emitter should have relative cartesian and polar coordinates
    assert isinstance(src.coordinates_relative_cartesian, dict)
    assert len(src.coordinates_relative_cartesian) == 1
    assert isinstance(src.coordinates_relative_polar, dict)
    assert len(src.coordinates_relative_cartesian) == 1


@pytest.mark.parametrize(
    "candidate_position,expected_idxs",
    [
        (np.array([0.6, 0.6, 0.6]), np.array([0])),
        (np.array([[0.6, 0.6, 0.6], [-0.6, -0.4, -0.3]]), np.array([0, 3])),
        (
            np.array([[1.2, 1.3, 1.3], [1.2, 1.3, 1.3], [-0.1, -0.1, -0.6]]),
            np.array([1, 1, 4]),
        ),
    ],
)
def test_get_nearest_source_idx(candidate_position, expected_idxs):
    ws = WorldStateSOFA(sofa=utils_tests.TEST_RESOURCES / "daga_foa.sofa")

    # Define some arbitrary source positions
    source_positions = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 1.5],
            [-0.5, -0.4, -0.3],
            [0.0, 0.0, -0.5],
        ]
    )

    # Hijack the method so that we return the desired source positions
    #  **not** the ones that are actually defined in the .sofa file
    ws.get_source_positions = Mock(return_value=source_positions)
    actual_idxs = ws.get_nearest_source_idx(candidate_position)

    # Should return the expected idxs
    assert np.array_equal(actual_idxs, expected_idxs)


@pytest.mark.parametrize("sofa_path", ["daga_foa.sofa", "metu_foa.sofa"])
@pytest.mark.parametrize("n_emitters", range(1, 4))
def test_simulate(sofa_path: str, n_emitters: int):
    # Create the WorldState
    ws = WorldStateSOFA(
        sofa=utils_tests.TEST_RESOURCES / sofa_path,
        mic_alias="tester",
        sample_rate=22050,
    )
    ws.clear_emitters()

    # Add some emitters in
    for _ in range(n_emitters):
        ws.add_emitter(keep_existing=True)

    # Do the simulation and grab the IRs
    ws.simulate()
    irs = ws.irs["tester"]
    assert isinstance(irs, np.ndarray)

    # Expecting FOA for this SOFA file
    n_ch, n_emitters_expected, n_samples = irs.shape
    assert n_ch == 4
    assert n_emitters_expected == n_emitters
    assert n_samples > 1

    # IRs should be resampled to desired rate
    with ws.sofa() as sofa:
        orig_sr = int(sofa.getVariableValue("Data.SamplingRate"))
        orig_ir = np.array(sofa.getDataIR().data)
        orig_n_samples = orig_ir.shape[-1]
    expected_n_samples = round(orig_n_samples * ws.sample_rate / orig_sr)
    assert n_samples == expected_n_samples


@pytest.mark.parametrize(
    "duration,max_speed,temporal_resolution",
    [
        # high velocity, small duration + resolution
        (0.5, 2.0, 1.0),
        # small resolution, high duration + velocity
        (5.0, 2.0, 1.0),
    ],
)
@pytest.mark.parametrize("shape", ["linear", "semicircular", None])
@pytest.mark.parametrize("sofa_path", ["daga_foa.sofa", "metu_foa.sofa"])
def test_define_trajectory(
    duration, max_speed, temporal_resolution, shape, sofa_path: str
):
    # Define the worldstate with the given .sofa file
    ws = WorldStateSOFA(
        sofa=utils_tests.TEST_RESOURCES / sofa_path,
    )

    # Define the trajectory
    trajectory = ws.define_trajectory(
        duration=duration,
        velocity=max_speed,
        resolution=temporal_resolution,
        shape=shape,
    )
    assert isinstance(trajectory, np.ndarray)

    # Check the shape: expecting (n_points, xyz == 3)
    n_points_actual, n_coords = trajectory.shape
    assert n_coords == 3
    assert n_points_actual >= 2

    # Check that speed constraints are never violated between points
    deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    max_segment_distance = max_speed / temporal_resolution
    assert np.all(deltas <= max_segment_distance + 1e-5)

    # Check distance between starting and ending point
    total_distance = np.linalg.norm(trajectory[-1, :] - trajectory[0, :])
    assert total_distance <= (max_speed * duration)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (dict(duration=5, shape="asdf"), ValueError),
        (
            dict(
                duration=5,
                starting_position=[0.5, 0.5, 0.5],
                velocity=0.01,
                resolution=100,
            ),
            ValueError,
        ),
        (
            dict(duration=5, velocity=0.01, resolution=100, max_place_attempts=10),
            ValueError,
        ),
    ],
)
def test_define_trajectory_invalid(kwargs, error, metu_space: WorldStateSOFA):
    with pytest.raises(error):
        _ = metu_space.define_trajectory(**kwargs)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # Not a valid trajectory
        (
            dict(
                trajectory=np.array([0]),
                max_distance=None,
                step_distance=None,
                n_points=None,
            ),
            False,
        ),
        # Number of points in trajectory does not match n_points
        (
            dict(
                trajectory=np.array([[0, 0, 0], [0, 0, 0]]),
                max_distance=None,
                step_distance=None,
                n_points=3,
            ),
            False,
        ),
        # Max distance exceeds constraint
        (
            dict(
                trajectory=np.array([[0, 0, 0], [100, 100, 100]]),
                max_distance=1,
                step_distance=1,
                n_points=2,
            ),
            False,
        ),
        # Max distance is fine, but step is too far
        (
            dict(
                trajectory=np.array([[0, 0, 0], [2, 2, 2], [0.5, 0.5, 0.5]]),
                max_distance=100,
                step_distance=0.5,
                n_points=3,
            ),
            False,
        ),
    ],
)
def test_validate_trajectory(kwargs, expected, metu_space: WorldStateSOFA):
    actual = metu_space._validate_trajectory(**kwargs)
    assert actual == expected


@pytest.mark.parametrize(
    "func_name",
    ["clear_microphone", "clear_microphones", "add_microphone", "add_microphones"],
)
def test_not_implemented_funcs(func_name: str, metu_space: WorldStateSOFA):
    func = getattr(metu_space, func_name)
    with pytest.raises(NotImplementedError):
        if func_name == "clear_microphone":
            func("asdf")
        else:
            func()


@pytest.mark.parametrize("sofa_name", ["metu_foa.sofa", "daga_foa.sofa"])
def test_to_dict(sofa_name):
    # Create the worldstate and add a few emitters in
    ws = WorldStateSOFA(
        sofa=utils_tests.TEST_RESOURCES / sofa_name,
        sample_rate=22050,
        mic_alias="tester",
    )
    for _ in range(4):
        ws.add_emitter(keep_existing=True)
    # Add another emitter in with the same alias so it gets nested
    ws.add_emitter(alias="src000", keep_existing=True)

    # Output the dictionary
    out_dict = ws.to_dict()

    # Back to a worldstate
    in_dict = WorldState.from_dict(out_dict)

    # Compare the two worldstates before/after serialisation: should be equal
    assert in_dict == ws


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "backend": "SOFA",
            "sofa": str(utils_tests.TEST_RESOURCES / "metu_foa.sofa"),
            "sample_rate": 22050,
            "emitters": {
                "src000": [[1.5, 1.5, -0.5], [-1.5, 0.5, 0.5]],
                "src001": [[1.0, 0.5, 0.0]],
                "src002": [[1.0, 1.5, 0.5]],
                "src003": [[1.5, -0.5, 0.0]],
            },
            "emitter_sofa_idxs": {
                "src000": [181, 203],
                "src001": [4],
                "src002": [243],
                "src003": [202],
            },
            "microphones": {
                "tester": {
                    "name": "monocapsule",
                    "micarray_type": "MonoCapsule",
                    "is_spherical": False,
                    "channel_layout_type": "mic",
                    "n_capsules": 1,
                    "capsule_names": ["mono"],
                    "coordinates_absolute": [[0.0, 0.0, 0.0]],
                    "coordinates_center": [0.0, 0.0, 0.0],
                }
            },
            "metadata": {
                "bounds": [[-1.5, -1.5, -1.0], [1.5, 1.5, 1.0]],
                "Conventions": "SOFA",
                "Version": "2.1",
                "SOFAConventions": "SingleRoomSRIR",
                "SOFAConventionsVersion": "1.0",
                "APIName": "pysofaconventions",
                "APIVersion": "0.1.5",
                "AuthorContact": "chris.ick@nyu.edu",
                "Organization": "Music and Audio Research Lab - NYU",
                "License": "Use whatever you want",
                "DataType": "FIR",
                "DateCreated": "Thu Apr 11 19:39:03 2024",
                "DateModified": "Thu Apr 11 19:39:03 2024",
                "Title": "METU-SPARG - classroom",
                "RoomType": "shoebox",
                "DatabaseName": "METU-SPARG",
                "ListenerShortName": "em32",
                "RoomShortName": "classroom",
                "Comment": "N/A",
            },
        }
    ],
)
def test_from_dict(input_dict):
    wstate = WorldState.from_dict(input_dict)
    assert isinstance(wstate, WorldStateSOFA)
    # Should have the correct number of emitters and microphones
    assert wstate.num_emitters == 5
    assert len(wstate.microphones) == 1


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "backend": "SOFA",
            "sofa": str(utils_tests.TEST_RESOURCES / "metu_foa.sofa"),
            "emitters": {},
            "microphones": {},
            "metadata": {},
        }
    ],
)
def test_from_invalid_dict(input_dict):
    # Missing some input keys
    with pytest.raises(KeyError):
        _ = WorldState.from_dict(input_dict)
