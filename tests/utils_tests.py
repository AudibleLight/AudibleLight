#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, constants used across all tests. Note: NOT THE SAME as `test_utils.py`!"""

import os

from audiblelight import utils

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
