#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, constants used across all tests. Note: NOT THE SAME as `test_utils.py`!"""

import os
from dataclasses import dataclass

import numpy as np

from audiblelight import utils
from audiblelight.micarrays import MicArray

TEST_RESOURCES = utils.get_project_root() / "tests/test_resources"
SOUNDEVENT_DIR = TEST_RESOURCES / "soundevents"
# Use tap for background audio
BACKGROUND_DIR = SOUNDEVENT_DIR / "waterTap"
MESH_DIR = TEST_RESOURCES / "meshes"
OYENS_PATH = MESH_DIR / "Oyens.glb"
OYENS_WAYPOINTS_PATH = MESH_DIR / "Oyens_waypoints.json"

TEST_MESHES = [MESH_DIR / glb for glb in MESH_DIR.glob("*.glb")]
TEST_AUDIOS = sorted(
    [
        os.path.join(xs, x)
        for xs in utils.list_deepest_directories(SOUNDEVENT_DIR)
        for x in os.listdir(xs)
        if x.endswith((".wav", ".mp3"))
    ]
)
TEST_MUSICS = [i for i in TEST_AUDIOS if "music" in i]

METU_SOFA_URL = "https://drive.google.com/uc?id=1zamCd6OR6Tr5M40RdDhswYbT1wbGo2ZO"
METU_SOFA_PATH = TEST_RESOURCES / "metu_foa.sofa"


@dataclass(eq=False)
class CubeMic(MicArray):
    """
    Custom MicArray class not defined inside `audiblelight.micarrays.MICARRAY_LIST`.
    """

    name: str = "cube"
    is_spherical = False
    # layout can be either FOA or MIC
    channel_layout_type = "mic"

    @property
    def coordinates_polar(self) -> np.ndarray:
        # Azimuth, elevation, radius
        return np.array(
            [
                [45, 30, 0.5],
                [135, 30, 0.5],
                [-135, 30, 0.5],
                [-45, 30, 0.5],
                [45, -30, 0.5],
                [135, -30, 0.5],
                [-135, -30, 0.5],
                [-45, -30, 0.5],
            ]
        )

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        # Defined automatically, just included here for reference
        return utils.polar_to_cartesian(self.coordinates_polar)

    @property
    def capsule_names(self) -> list[str]:
        # Front-left upper, back-left upper, back-right upper, front-right upper
        # Front-left lower, back-left lower, back-right lower, front-right lower
        return ["FLU", "BLU", "BRU", "FRU", "FLL", "BLL", "BRL", "FRL"]
