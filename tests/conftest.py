#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures used across all tests"""
from typing import Callable

import pytest

from audiblelight.core import Scene
from audiblelight.worldstate import WorldState
from tests import utils_tests


@pytest.fixture(scope="function")
def oyens_space() -> WorldState:
    """Returns a WorldState object with the Oyens mesh (Gibson)"""
    space = WorldState(
        utils_tests.OYENS_PATH,
        empty_space_around_emitter=0.2,  # all in meters
        empty_space_around_mic=0.1,  # all in meters
        empty_space_around_surface=0.2,  # all in meters
    )
    return space


@pytest.fixture(scope="function")
def oyens_scene_no_overlap() -> Scene:
    """Returns a scene object with the Oyens mesh (Gibson), that doesn't allow for overlapping Events"""
    # Create a dummy scene
    sc = Scene(
        duration=50,
        mesh_path=utils_tests.OYENS_PATH,
        # Use the default distribution for everything
        # event_start_dist=stats.uniform(0, 10),
        # event_duration_dist=stats.uniform(0, 10),
        # event_velocity_dist=stats.uniform(0, 10),
        # event_resolution_dist=stats.uniform(0, 10),
        # snr_dist=stats.norm(5, 1),
        fg_path=utils_tests.SOUNDEVENT_DIR,
        max_overlap=1,  # no overlapping sound events allowed
    )
    sc.add_microphone(microphone_type="ambeovr")
    return sc


@pytest.fixture
def oyens_scene_factory() -> Callable:
    def _factory():
        sc = Scene(
            duration=50,
            mesh_path=utils_tests.OYENS_PATH,
            # event_start_dist=stats.uniform(0, 10),
            # event_duration_dist=stats.uniform(0, 10),
            # event_velocity_dist=stats.uniform(0, 10),
            # event_resolution_dist=stats.uniform(0, 10),
            # snr_dist=stats.norm(5, 1),
            fg_path=utils_tests.SOUNDEVENT_DIR,
            max_overlap=1,
        )
        sc.add_microphone(microphone_type="ambeovr")
        return sc

    return _factory
