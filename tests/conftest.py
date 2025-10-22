#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures used across all tests"""

import gc
from typing import Callable

import pytest
from tqdm import tqdm

from audiblelight.core import Scene
from audiblelight.worldstate import WorldState
from tests import utils_tests

tqdm.monitor_interval = 0


@pytest.fixture(scope="function")
def oyens_space() -> WorldState:
    """Returns a WorldState object with the Oyens mesh (Gibson)"""
    space = WorldState(
        utils_tests.OYENS_PATH,
        add_to_context=True,  # update worldstate with every addition
        empty_space_around_emitter=0.2,  # all in meters
        empty_space_around_mic=0.1,  # all in meters
        empty_space_around_surface=0.2,  # all in meters
        waypoints_json=utils_tests.OYENS_WAYPOINTS_PATH,
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
        bg_path=utils_tests.BACKGROUND_DIR,
        max_overlap=1,  # no overlapping sound events allowed
        state_kwargs=dict(waypoints_json=utils_tests.OYENS_WAYPOINTS_PATH),
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
            state_kwargs=dict(waypoints_json=utils_tests.OYENS_WAYPOINTS_PATH),
        )
        sc.add_microphone(microphone_type="ambeovr")
        return sc

    return _factory


@pytest.fixture(autouse=True)
def run_gc_after_test():
    yield
    gc.collect()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

    # Remove exception info, since it causes excessive memory usage and workers crash
    if call.excinfo and isinstance(call.excinfo, pytest.ExceptionInfo):
        call.excinfo = None
