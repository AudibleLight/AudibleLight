#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py with SOFA backend"""


import numpy as np
import pytest

from audiblelight import synthesize, utils
from audiblelight.augmentation import LowpassFilter, Phaser
from audiblelight.core import Scene
from audiblelight.event import Event
from audiblelight.worldstate import WorldStateSOFA
from tests import utils_tests


@pytest.mark.parametrize(
    "kwargs",
    [
        # Explicit event keywords and filepath, but no emitter keywords
        dict(filepath=utils_tests.SOUNDEVENT_DIR / "music/001666.mp3", snr=5),
        # Explicit event and emitter keywords, but no filepath (will be randomly sampled)
        dict(
            position=np.array([-0.5, -0.5, 0.5]),
            polar=False,
            duration=5,
            event_start=5,
            scene_start=5,
            snr=5,
        ),
        # no kwargs
        dict(),
        # polar position
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "maleSpeech/93853.wav",
            polar=True,
            position=[90, 0.0, 0.5],
            scene_start=0.0,
            augmentations=[LowpassFilter, Phaser()],
        ),
    ],
)
def test_add_event_static(kwargs, metu_scene_no_overlap: Scene):
    assert isinstance(metu_scene_no_overlap.state, WorldStateSOFA)

    # Try adding some events
    metu_scene_no_overlap.add_event(**kwargs, event_type="static", alias="test_event")

    # Get the event
    ev = metu_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert not ev.is_moving
    assert ev.has_emitters
    assert len(ev) == 1

    # Check that the starting and end time of the event is within temporal bounds for the scene
    assert ev.scene_start >= 0
    assert ev.scene_end < metu_scene_no_overlap.duration

    # If we've passed in a custom position for the emitter, ensure that this is set correctly
    desired_position = kwargs.get("position", None)
    is_polar = kwargs.get("polar", False)

    # Actual position of event should be nearest neighbour to desired position
    if desired_position is not None:
        # Polar: convert to cartesian WRT mic
        if is_polar:
            desired_position = (
                metu_scene_no_overlap.get_microphone("mic000").coordinates_center
                + utils.polar_to_cartesian(desired_position)[0]
            )

        # Check that position of event is nearest neighbour of desired position
        cand_positions = metu_scene_no_overlap.state.get_source_positions()
        distances = np.linalg.norm(cand_positions - desired_position, axis=1)
        nearest_neighbour = cand_positions[np.argmin(distances), :]
        assert np.allclose(nearest_neighbour, ev.get_emitter(0).coordinates_absolute)

        # Check distance between desired and actual position
        #  Should be below a small threshold
        assert np.min(distances) <= 0.2

    # Check all overrides passed correctly to the event class
    #  When we're using a random file, we cannot check these variables as they might have changed
    #  due to cases where the duration of the random file is shorter than the passed value (5 seconds)
    if kwargs.get("filepath") is not None:
        for override_key, override_val in kwargs.items():
            if hasattr(ev, override_key):
                attr = getattr(ev, override_key)
                # Skip nested keys
                if isinstance(attr, (list, np.ndarray, tuple, set)):
                    continue
                assert attr == override_val


@pytest.mark.parametrize("n_events", range(1, 4))
def test_synthesise_with_sofa(n_events, metu_scene_no_overlap: Scene):
    # Clear everything out
    metu_scene_no_overlap.clear_events()
    metu_scene_no_overlap.clear_ambience()

    # Add some events
    for _ in range(n_events):
        metu_scene_no_overlap.add_event_static()

    # Add some ambience
    metu_scene_no_overlap.add_ambience(noise="white", channels=4)

    # Render the scene audio
    synthesize.validate_scene(metu_scene_no_overlap)
    synthesize.render_audio_for_all_scene_events(metu_scene_no_overlap)

    # Now, try generating the full scene audio
    synthesize.generate_scene_audio_from_events(metu_scene_no_overlap)
    assert isinstance(metu_scene_no_overlap.audio["mic000"], np.ndarray)

    # Audio should have the expected number of channels and duration
    channels, duration = metu_scene_no_overlap.audio["mic000"].shape
    expected = round(metu_scene_no_overlap.sample_rate * metu_scene_no_overlap.duration)
    assert duration == expected
