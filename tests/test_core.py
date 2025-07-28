#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py"""

import os
from pathlib import Path

import pytest
import numpy as np

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.event import Event
from audiblelight.micarrays import MicArray, AmbeoVR
from audiblelight.worldstate import Emitter


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
        # Test 4: no path, no kwargs
        (
            None, None, None
        ),
    ]
)
def test_add_event(filepath: str, emitter_kws, event_kws, oyens_scene_no_overlap: Scene):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        filepath=filepath,
        alias="test_event",
        emitter_kwargs=emitter_kws,
        event_kwargs=event_kws
    )
    # Should be added to ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)

    # If we've passed in a custom position for the emitter, ensure that this is set correctly
    if isinstance(emitter_kws, dict):
        desired_position = emitter_kws.get("position", None)
        if desired_position is not None:
            assert np.array_equal(desired_position, ev.start_coordinates_absolute)

    # Check all overrides passed correctly to the event class
    #  When we're using a random file, we cannot check these variables as they might have changed
    #  due to cases where the duration of the random file is shorter than the passed value (5 seconds)
    if filepath is not None and event_kws is not None:
        for override_key, override_val in event_kws.items():
            assert getattr(ev, override_key) == override_val

    # Check attributes that we will be adding into the event in its __init__ call based on the kwargs
    for attr_ in ["event_end", "start_coordinates_absolute", "end_coordinates_relative_polar"]:
        assert hasattr(ev, attr_)


@pytest.mark.parametrize(
    "new_event_audio,event_kwargs,raises",
    [
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(scene_start=6.0, duration=1.0), ValueError),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(scene_start=9.0, duration=10.0), ValueError),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(scene_start=3.0, duration=10.0), ValueError),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(sample_rate=12345), ValueError),    # sample rate different to state
    ]
)
def test_add_bad_event(new_event_audio, event_kwargs, raises, oyens_scene_no_overlap: Scene):
    """
    Test adding an event that should be rejected
    """
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        filepath=SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        event_kwargs=dict(
            scene_start=5.0,
            duration=5.0
        )
    )
    # Add the tester event in: should raise an error
    with pytest.raises(raises):
        oyens_scene_no_overlap.add_event(
            filepath=new_event_audio,
            alias="bad_event",
            emitter_kwargs=dict(keep_existing=True),
            event_kwargs=event_kwargs
        )
    # Should not be added to the dictionary
    assert len(oyens_scene_no_overlap.events) == 1
    with pytest.raises(KeyError):
        _ = oyens_scene_no_overlap.get_event("bad_event")


@pytest.mark.parametrize(
    "new_event_audio,new_event_kws",
    [
        (SOUNDEVENT_DIR / "music/000010.mp3", dict()),    # no custom event_start/duration, should be set automatically
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(event_start=15, duration=20)),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(duration=5.0)),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(scene_start=10.0, duration=5.0, event_start=5.0)),   # no overlap
    ]
)
def test_add_acceptable_event(new_event_audio, new_event_kws, oyens_scene_no_overlap: Scene):
    """
    Test adding an acceptable event to a scene that already has events; should not be rejected
    """
    # Add the dummy event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        filepath=SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=dict(
            scene_start=5.0,
            duration=5.0
        )
    )
    # Add the tester event in: should not raise any errors
    oyens_scene_no_overlap.add_event(
        filepath=new_event_audio,
        alias="good_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=new_event_kws
    )
    # Should be able to access the event class
    assert len(oyens_scene_no_overlap.events) == 2
    ev = oyens_scene_no_overlap.get_event("good_event")
    assert isinstance(ev, Event)
    assert ev.filepath == new_event_audio
    # Should have two emitters in the ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 2


@pytest.mark.parametrize(
    "fg_path,raises",
    [
        ([SOUNDEVENT_DIR / "music"], False),    # this folder has some audio files inside it, so it's all good
        (None, ValueError),    # as if we've not provided `fp_path` to `Scene.__init__`
        ([utils.get_project_root() / "tests"], FileNotFoundError)    # no audio files inside this folder!
    ],
)
def test_get_random_foreground_audio(fg_path: str, raises, oyens_scene_no_overlap: Scene):
    setattr(oyens_scene_no_overlap, "fg_category_paths", fg_path)
    if not raises:
        out = oyens_scene_no_overlap._get_random_foreground_audio()
        assert str(out).endswith(utils.AUDIO_EXTS)
        assert os.path.isfile(out)
        assert isinstance(out, Path)
    else:
        with pytest.raises(raises):
            _ = oyens_scene_no_overlap._get_random_foreground_audio()


@pytest.mark.parametrize(
    "mic_arrays",
    [
        (dict(microphone_type="ambeovr", position=np.array([-0.5, -0.5, 0.5]), alias="mic000")),
        (dict(microphone_type="ambeovr", alias="mic000")),
    ]
)
def test_add_mic_arrays_to_state(mic_arrays, oyens_scene_no_overlap: Scene):
    # Remove anything added to the space
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.state.clear_microphones()
    oyens_scene_no_overlap.add_microphone(**mic_arrays)
    # Try and get the microphone
    mic = oyens_scene_no_overlap.get_microphone("mic000")
    assert issubclass(type(mic), MicArray)
    # Should have attributes set
    for attr_ in ["coordinates_absolute", "n_capsules"]:
        assert hasattr(mic, attr_)
    # Position should be set properly if we've passed this
    if isinstance(mic_arrays, dict) and "position" in mic_arrays:
        assert np.array_equal(mic.coordinates_center, mic_arrays["position"])
    # Number of listeners should all be set OK in the ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_listener_count() == 4


def test_get_funcs(oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.add_event()
    # Test all the get functions
    mic = oyens_scene_no_overlap.get_microphone("mic000")
    assert issubclass(type(mic), MicArray)
    emitter_list = oyens_scene_no_overlap.get_emitters("event000")
    assert isinstance(emitter_list, list)
    emitter = oyens_scene_no_overlap.get_emitter("event000", 0)
    assert isinstance(emitter, Emitter)
    event = oyens_scene_no_overlap.get_event("event000")
    assert isinstance(event, Event)
    event2 = oyens_scene_no_overlap["event000"]
    assert isinstance(event2, Event)

    # Trying to get event that doesn't exist will raise an error
    with pytest.raises(KeyError):
        oyens_scene_no_overlap.get_event("not_existing")


def test_clear_event(oyens_scene_no_overlap: Scene):
    # Add an event
    oyens_scene_no_overlap.add_event(alias="remover")
    assert len(oyens_scene_no_overlap.events) == 1
    assert len(oyens_scene_no_overlap.state.emitters) == 1
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 1

    # Remove the event and check all has been removed
    oyens_scene_no_overlap.clear_event(alias="remover")
    assert len(oyens_scene_no_overlap.events) == 0
    assert len(oyens_scene_no_overlap.state.emitters) == 0
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 0

    # Trying to remove event that doesn't exist will raise an error
    with pytest.raises(KeyError):
        oyens_scene_no_overlap.clear_event("not_existing")

    # By default, the fixture has one mic with four capsules, so check these
    assert len(oyens_scene_no_overlap.state.microphones) == 1
    assert oyens_scene_no_overlap.state.ctx.get_listener_count() == 4


@pytest.mark.parametrize(
    "noise, filepath",
    [
        ("white", None),
        (2.0, None),
        (None, utils.get_project_root() / "tests/test_resources/soundevents/waterTap/95709.wav")
    ]
)
def test_add_ambience(noise, filepath, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_ambience()
    oyens_scene_no_overlap.add_ambience(noise=noise, filepath=filepath, alias="tester")
    amb = oyens_scene_no_overlap.get_ambience("tester")
    ambience_audio = amb.load_ambience()
    assert isinstance(ambience_audio, np.ndarray)
    expected_duration = oyens_scene_no_overlap.duration * oyens_scene_no_overlap.sample_rate
    assert ambience_audio.shape == (4, expected_duration)
