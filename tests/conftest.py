#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures used across all tests"""

import pytest

from audiblelight.worldstate import WorldState
from audiblelight import utils

OYENS_MESH = utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb"


@pytest.fixture(scope="function")
def oyens_space() -> WorldState:
    """Returns a WorldState object with the Oyens mesh (Gibson)"""
    space = WorldState(
        OYENS_MESH,
        empty_space_around_emitter=0.2,    # all in meters
        empty_space_around_mic=0.1,    # all in meters
        empty_space_around_surface=0.2    # all in meters
    )
    return space
