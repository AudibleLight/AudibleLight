#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/event.py"""

import os
from typing import Optional

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.event import Event

SOUNDEVENT_DIR = utils.get_project_root() / "tests/test_resources/soundevents"
TEST_AUDIO_DIRS = utils.list_deepest_directories(SOUNDEVENT_DIR)
TEST_AUDIOS = sorted([os.path.join(xs, x) for xs in TEST_AUDIO_DIRS for x in os.listdir(xs) if x.endswith((".wav", ".mp3"))])


@pytest.mark.parametrize("audio_fpath", TEST_AUDIOS[:5])
def test_create_static_event(audio_fpath: str, oyens_space):
    oyens_space.add_microphone()
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(audio_fpath, "test_event", emitters=emitter)
    assert not ev.is_moving
    # Coordinates should be set properly
    assert np.array_equal(ev.start_coordinates_absolute, emitter[0].coordinates_absolute)
    assert np.array_equal(ev.end_coordinates_absolute, emitter[0].coordinates_absolute)
    # Should be able to create a dictionary
    ev_dict = ev.to_dict()
    for k in ["alias", "duration", "start_coordinates", "end_coordinates", "filepath"]:
        assert k in ev_dict.keys()


@pytest.mark.parametrize(
    "audio_fpath,duration,start_time",
    [
        # These are all music audio files, which we use as they're long
        (TEST_AUDIOS[6], 0.5, 0.5),
        (TEST_AUDIOS[7], 1.0, 1.0),
        (TEST_AUDIOS[8], None, 1.0),
        (TEST_AUDIOS[6], 0.5, None),
        (TEST_AUDIOS[7], None, None),
        (TEST_AUDIOS[8], 2.0, 5.0),
    ]
)
def test_load_audio(audio_fpath: str, duration: Optional[float], start_time: Optional[float], oyens_space):
    # Create a dummy event
    oyens_space.add_microphone()
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(
        audio_fpath,
        "test_event",
        emitters=emitter,
        duration=duration,
        event_start=start_time,
        sample_rate=utils.SAMPLE_RATE
    )
    # Try and load the audio
    audio = ev.load_audio(ignore_cache=True)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1    # should be mono
    assert ev._audio_loaded
    # Try and load the audio again, should be cached
    audio2 = ev.load_audio(ignore_cache=False)
    assert np.array_equal(audio, audio2)
    # If we've passed in a custom duration, this should be respected
    if duration is not None:
        assert len(audio) / utils.SAMPLE_RATE == duration
        # We should set the end time correctly
        if start_time is not None:
            assert ev.event_end == start_time + duration


@pytest.mark.parametrize(
    "duration,expected",
    [
        (None, 30.0),    # No explicit duration: use full duration
        (5000, 30.0),    # too long duration, use full duration
        (5.0, 5.0)    # acceptable duration, use it
    ]
)
def test_parse_duration(duration: float, expected: float, oyens_space):
    # Create a dummy event
    oyens_space.add_microphone()
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(
        filepath=SOUNDEVENT_DIR / "music/007527.mp3",    # duration of this audio is roughly 30 seconds
        alias="test_event",
        emitters=emitter,
        duration=duration,
        sample_rate=utils.SAMPLE_RATE
    )
    parsed_duration = round(ev._parse_duration(duration))    # should be 30 seconds
    assert np.isclose(parsed_duration, expected)

