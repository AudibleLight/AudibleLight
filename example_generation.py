#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate a simple scene with moving, static, and ambient sound events and render to audio and JSON."""

import os

from scipy import stats

from audiblelight import utils
from audiblelight.core import Scene

# OUTPUT DIRECTORY
OUTFOLDER = utils.get_project_root() / 'spatial_scenes'
if not os.path.isdir(OUTFOLDER):
    os.makedirs(OUTFOLDER)

# PATHS
FG_FOLDER = utils.get_project_root() / "tests/test_resources/soundevents"
MESH_PATH = utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb"  # Mesh can be a "building"

# SCENE SETTINGS
DURATION = 30.0  # seconds
MIC_ARRAY_NAME = 'tetra'
N_STATIC_EVENTS = 4
N_MOVING_EVENTS = 1
MAX_OVERLAP = 3

# SCENE-WIDE DISTRIBUTIONS
SCENE_TIME_DIST = stats.uniform(0.0, DURATION - 1)    # controls when events start in the scene
EVENT_DURATION_DIST = stats.uniform(0.5, 4.0)
EVENT_VELOCITY_DIST = stats.uniform(0.1, 1.5)  # meters per second
SNR_DIST = stats.uniform(6, 30)
REF_DB = -50


def mvp() -> None:
    """
    Creates an example generation with arguments set above
    """
    sc = Scene(
        duration=DURATION,
        mesh_path=MESH_PATH,
        scene_start_dist=SCENE_TIME_DIST,
        event_duration_dist=EVENT_DURATION_DIST,
        event_velocity_dist=EVENT_VELOCITY_DIST,
        snr_dist=SNR_DIST,
        fg_path=FG_FOLDER,
        max_overlap=MAX_OVERLAP,
        ref_db=REF_DB
    )

    # Add an ambeoVR microphone to the scene
    sc.add_microphone(microphone_type="ambeovr", alias=MIC_ARRAY_NAME)

    # Add required sources to the scene
    for _ in range(N_STATIC_EVENTS):
        sc.add_event(event_type="static", emitter_kwargs=dict(keep_existing=True))

    for _ in range(N_MOVING_EVENTS):
        sc.add_event(event_type="moving",)

    # Add some white noise as ambience
    sc.add_ambience(noise="white")

    # Also add an audio file as ambience
    #  This will be tiled to match the required duration and number of channels
    sc.add_ambience(
        filepath=utils.get_project_root() / "tests/test_resources/soundevents/waterTap/95709.wav"
    )

    # Generate the audio
    sc.generate(audio_path=OUTFOLDER / "audio_out.wav", metadata_path=OUTFOLDER / "metadata_out.json")


if __name__ == "__main__":
    mvp()
