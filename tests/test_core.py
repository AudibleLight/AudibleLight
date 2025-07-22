#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py"""

import pytest
import numpy as np

from scipy import stats

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.event import Event


SOUNDEVENT_DIR = utils.get_project_root() / "tests/test_resources/soundevents"
OYENS_PATH = utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb"


# @pytest.mark.parametrize(
#     "mesh_path,mic_arrays"
# )
# def test_create_scene(mesh_path, mic_arrays, ):
#     pass


@pytest.mark.parametrize(
    "filepath,emitter_kws,event_kws",
    [
        # Test 1: explicitly define a filepath, emitter keywords, and event keywords (overrides)
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
            dict(
                position=np.array([-0.5, -0.5, 0.5]),
                polar=False,
                ensure_direct_path=False
            ),
            dict(
                duration=5,
                event_start=5,
                scene_start=5
            )
        ),
        # Test 2: explicit event keywords and filepath, but no emitter keywords
        (
            SOUNDEVENT_DIR / "music/001666.mp3",
            None,
            dict(
                snr=5,
                spatial_velocity=5
            )
        ),
        # Test 3: explicit event and emitter keywords, but no filepath (will be randomly sampled)
        (
            None,
            dict(
                position=np.array([-0.5, -0.5, 0.5]),
                polar=False,
                ensure_direct_path=False
            ),
            dict(
                duration=5,
                event_start=5,
                scene_start=5,
                snr=5,
                spatial_velocity=5
            )
        ),
    ]
)
def test_add_event(filepath: str, emitter_kws, event_kws, oyens_scene: Scene):
    # Add the event in
    oyens_scene.clear_events()
    oyens_scene.add_event(
        filepath=filepath,
        alias="test_event",
        emitter_kwargs=emitter_kws,
        event_kwargs=event_kws
    )

    # Get the event
    ev = oyens_scene.get_event("test_event")
    assert isinstance(ev, Event)

    # If we've passed in a custom position for the emitter, ensure that this is set correctly
    if isinstance(emitter_kws, dict):
        desired_position = emitter_kws.get("position", None)
        if desired_position is not None:
            assert np.array_equal(desired_position, ev.start_coordinates_absolute)

    # Check all overrides passed correctly to the event class
    for override_key, override_val in event_kws.items():
        assert getattr(ev, override_key) == override_val

    # Check attributes that we will be adding into the event in its __init__ call based on the kwargs
    for attr_ in ["event_end", "start_coordinates_absolute", "end_coordinates_relative_polar"]:
        assert hasattr(ev, attr_)


@pytest.mark.parametrize(
    "new_event_audio,new_event_start,new_event_duration",
    [
        (SOUNDEVENT_DIR / "music/000010.mp3", 6.0, 1.0),
        (SOUNDEVENT_DIR / "music/000010.mp3", 9.0, 10.0),
        (SOUNDEVENT_DIR / "music/000010.mp3", 3.0, 10.0),
    ]
)
def test_add_overlapping_new_event(new_event_audio, new_event_start, new_event_duration, oyens_scene: Scene):
    """
    Test adding an event that overlaps with existing ones: should be rejected
    """
    # Add the event in
    oyens_scene.clear_events()
    oyens_scene.add_event(
        filepath=SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        event_kwargs=dict(
            scene_start=5.0,
            duration=5.0
        )
    )
    # Add the tester event in: should raise an error due to overlap
    with pytest.raises(ValueError):
        oyens_scene.add_event(
            filepath=new_event_audio,
            alias="bad_event",
            emitter_kwargs=dict(keep_existing=True),
            event_kwargs=dict(
                scene_start=new_event_start,
                duration=new_event_duration
            )
        )
    # Should not be added to the dictionary
    assert len(oyens_scene.events) == 1
    with pytest.raises(KeyError):
        _ = oyens_scene.get_event("bad_event")


@pytest.mark.parametrize(
    "new_event_audio,new_event_kws",
    [
        (SOUNDEVENT_DIR / "music/000010.mp3", dict()),    # no custom event_start/duration, should be set automatically
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(event_start=15, duration=20)),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(duration=5.0)),
    ]
)
def test_add_nonoverlapping_new_event(new_event_audio, new_event_kws, oyens_scene: Scene):
    """
    Test adding an acceptable event to a scene that already has events; should not be rejected
    """
    # Add the dummy event in
    oyens_scene.clear_events()
    oyens_scene.add_event(
        filepath=SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=dict(
            scene_start=5.0,
            duration=5.0
        )
    )
    # Add the tester event in: should not raise any errors
    oyens_scene.add_event(
        filepath=new_event_audio,
        alias="good_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=new_event_kws
    )
    # Should be able to access the event class
    assert len(oyens_scene.events) == 2
    ev = oyens_scene.get_event("good_event")
    assert isinstance(ev, Event)
    assert ev.filepath == new_event_audio
    # Should have two emitters in the ray-tracing engine
    assert oyens_scene.state.ctx.get_source_count() == 2
