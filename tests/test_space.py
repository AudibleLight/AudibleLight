#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/space.py"""

import os

import pytest
import numpy as np
from trimesh import Trimesh

from audiblelight.space import Space, load_mesh, repair_mesh
from audiblelight import utils


TEST_RESOURCES = utils.get_project_root() / "tests/test_resources"
TEST_MESHES = [TEST_RESOURCES / glb for glb in TEST_RESOURCES.glob("*.glb")]


@pytest.mark.parametrize("mesh_fpath", TEST_MESHES)
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
def test_load_mesh(mesh_fpath: str):
    loaded = load_mesh(mesh_fpath)
    assert isinstance(loaded, Trimesh)
    assert loaded.metadata["fpath"] == str(mesh_fpath)    # need both to be a string or we'll get TypeError
    assert loaded.units == utils.MESH_UNITS    # units should be in meters


@pytest.mark.parametrize("mesh_fpath,expected", [("iamnotafile", FileNotFoundError), (1234, TypeError)])
def test_load_broken_mesh(mesh_fpath: str, expected):
    with pytest.raises(expected):
        load_mesh(mesh_fpath)


@pytest.fixture(scope="module")
def oyens_space() -> Space:
    """Returns a Space object with the Oyens mesh (Gibson) and a single microphone"""
    oyens = os.path.join(utils.get_project_root(), "tests/test_resources/Oyens.glb")
    space = Space(
        oyens,
        mic_positions=[-0.5, -0.5, 0.5],
        min_distance_from_source=0.2,    # all in meters
        min_distance_from_mic=0.1,    # all in meters
        min_distance_from_surface=0.2    # all in meters
    )
    return space


@pytest.mark.parametrize(
    "microphone_positions",
    [
        np.array([[-0.5, -0.5, 0.5,]]),    # 1 mic setup
        np.array([[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5]])     # 2 mic setup
    ]
)
def test_place_microphones(microphone_positions):
    space = Space(TEST_MESHES[0], microphone_positions)
    # Should have the same number of mic positions we've passed in
    assert isinstance(space.mic_positions, np.ndarray)
    # All mic positions should be in XYZ format
    assert space.mic_positions.shape == (len(microphone_positions), 3)
    # Iterate over all mics
    for mic_position in space.mic_positions:
        # Should be inside the mesh
        assert space._is_point_inside_mesh(mic_position)


@pytest.mark.parametrize(
    "test_position,expected",
    [
        (np.array([-0.4, -0.5, 0.5]), False),    # Too close to mic
        (np.array([-0.5, -0.4, 0.5]), False),    # Too close to mic
        (np.array([-0.5, -0.5, 0.4]), False),    # Too close to mic
        (np.array([-0.8, -1.5, 0.2]), False),    # Too close to the surface
        (np.array([-0.1, -0.1, 0.6]), True),    # Fine!
        (np.array([0.5, 0.5, 0.5]), True)    # Also fine
    ]
)
def test_validate_positions(test_position: np.ndarray, expected: bool, oyens_space: Space):
    """Given a microphone with coordinates [-0.5, -0.5, 0.5], test whether test_position is valid"""
    assert oyens_space._validate_source_position(test_position) == expected


@pytest.mark.parametrize(
    "test_position,expected_shape",
    [
        (np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]), (1, 3)),    # 1: too close to mic, 2: fine
        (np.array([[0.5, 0.5, 0.5], [0.6, 0.4, 0.5]]), (1, 3)),    # 1: fine, 2: too close to source 1
    ]
)
def test_add_sources(test_position: np.ndarray, expected_shape: tuple[int], oyens_space: Space):
    # Add the sources in and check that the shape of the resulting array is what we expect
    oyens_space.add_sources(test_position)
    assert oyens_space.source_positions.shape == expected_shape


@pytest.mark.parametrize(
    "test_position,expected_shape",
    [
        (np.array([[0.1, 0.0, 0.0], [-0.2, 0.2, 0.2]]), (1, 3)),    # 1: too close to mic, 2: fine
        (np.array([[-0.2, 0.2, 0.2], [-0.2, 0.3, 0.2]]), (1, 3)),    # 1: fine, 2: too close to source 1
        (np.array([[-0.2, 0.2, 0.2], [0.2, -0.3, -0.2]]), (2, 3)),  # both fine
    ]
)
def test_add_sources_relative_to_mic(test_position: np.ndarray, expected_shape: tuple[int], oyens_space: Space):
    # Add the sources in and check that the shape of the resulting array is what we expect
    oyens_space.add_sources_relative_to_mic(test_position, 0)
    assert oyens_space.source_positions.shape == expected_shape


@pytest.mark.parametrize("num_rays", [1, 10, 100])
def test_calculate_weighted_average_ray_length(num_rays: int, oyens_space: Space):
    # Get a random valid point inside the mesh
    point = oyens_space.get_random_position()
    result = oyens_space.calculate_weighted_average_ray_length(point, num_rays=num_rays)
    # Validate output is positive float and finite (since rays should hit mesh)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result > 0


@pytest.mark.parametrize("test_num", range(5))
def test_get_random_position(test_num: int, oyens_space: Space):
    # For reproducible results
    utils.seed_everything(test_num)
    # Add some sources to the space
    oyens_space.add_random_sources(test_num)
    # Grab a random position
    random_point = oyens_space.get_random_position()
    # It should be valid (suitable distance from surfaces, inside mesh, away from mics/sources...)
    assert oyens_space._validate_source_position(random_point)
    assert random_point.shape == (3,)   # should be a 1D array of XYZ
