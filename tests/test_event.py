#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/event.py"""

from typing import Optional

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.event import Event
from tests import utils_tests


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_AUDIOS[:5])
def test_create_static_event(audio_fpath: str, oyens_space):
    oyens_space.add_microphone()
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(audio_fpath, "test_event", emitters=emitter)
    assert not ev.is_moving
    # Coordinates should be set properly
    assert np.array_equal(
        ev.start_coordinates_absolute, emitter[0].coordinates_absolute
    )
    assert np.array_equal(ev.end_coordinates_absolute, emitter[0].coordinates_absolute)
    # Should be able to create a dictionary
    ev_dict = ev.to_dict()
    for k in ["alias", "duration", "start_coordinates", "end_coordinates", "filepath"]:
        assert k in ev_dict.keys()


@pytest.mark.parametrize(
    "audio_fpath,duration,start_time",
    [
        # These are all music audio files, which we use as they're long
        (utils_tests.TEST_AUDIOS[6], 0.5, 0.5),
        (utils_tests.TEST_AUDIOS[7], 1.0, 1.0),
        (utils_tests.TEST_AUDIOS[8], None, 1.0),
        (utils_tests.TEST_AUDIOS[6], 0.5, None),
        (utils_tests.TEST_AUDIOS[7], None, None),
        (utils_tests.TEST_AUDIOS[8], 2.0, 5.0),
    ],
)
def test_load_audio(
    audio_fpath: str,
    duration: Optional[float],
    start_time: Optional[float],
    oyens_space,
):
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
        sample_rate=utils.SAMPLE_RATE,
    )
    # Try and load the audio
    audio = ev.load_audio(ignore_cache=True)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1  # should be mono
    assert ev.is_audio_loaded
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
        (None, 30.0),  # No explicit duration: use full duration
        (5000, 30.0),  # too long duration, use full duration
        (5.0, 5.0),  # acceptable duration, use it
    ],
)
def test_parse_duration(duration: float, expected: float, oyens_space):
    # Create a dummy event
    oyens_space.add_microphone()
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(
        filepath=utils_tests.SOUNDEVENT_DIR
        / "music/007527.mp3",  # duration of this audio is roughly 30 seconds
        alias="test_event",
        emitters=emitter,
        duration=duration,
        sample_rate=utils.SAMPLE_RATE,
    )
    parsed_duration = round(ev._parse_duration(duration))  # should be 30 seconds
    assert np.isclose(parsed_duration, expected)


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "alias": "test_event",
            "filename": "000010.mp3",
            "filepath": str(utils_tests.SOUNDEVENT_DIR / "music/000010.mp3"),
            "class_id": None,
            "class_label": None,
            "is_moving": False,
            "scene_start": 0.0,
            "scene_end": 29.976575963718822,
            "event_start": 0.0,
            "event_end": 29.976575963718822,
            "duration": 29.976575963718822,
            "snr": None,
            "sample_rate": 44100.0,
            "spatial_resolution": None,
            "spatial_velocity": None,
            "start_coordinates": [
                2.415245142454964,
                -4.396272957238952,
                1.1108193078909387,
            ],
            "end_coordinates": [
                2.415245142454964,
                -4.396272957238952,
                1.1108193078909387,
            ],
            "emitters": [
                {
                    "alias": "test_emitter",
                    "coordinates_absolute": [
                        2.415245142454964,
                        -4.396272957238952,
                        1.1108193078909387,
                    ],
                }
            ],
        },
        {
            "alias": "test_event",
            "filename": "000010.mp3",
            "filepath": str(utils_tests.SOUNDEVENT_DIR / "music/000010.mp3"),
            "class_id": None,
            "class_label": None,
            "is_moving": True,
            "scene_start": 5.0,
            "scene_end": 10.0,
            "event_start": 5.0,
            "event_end": 10.0,
            "duration": 5.0,
            "snr": 3.988549021563197,
            "sample_rate": 44100.0,
            "spatial_resolution": 1.9607707651309219,
            "spatial_velocity": 1.456467608999829,
            "start_coordinates": [
                0.6104437590653142,
                -0.6826065365865546,
                1.5575192418640027,
            ],
            "end_coordinates": [
                4.973968181221656,
                1.4143926984162998,
                1.3894620504871185,
            ],
            "emitters": [
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        0.6104437590653142,
                        -0.6826065365865546,
                        1.5575192418640027,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        1.0467962012809484,
                        -0.47290661308626913,
                        1.5407135227263142,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        1.4831486434965826,
                        -0.2632066895859837,
                        1.5239078035886258,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        1.9195010857122168,
                        -0.053506766085698265,
                        1.5071020844509375,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        2.355853527927851,
                        0.15619315741458717,
                        1.490296365313249,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        2.792205970143485,
                        0.3658930809148726,
                        1.4734906461755606,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        3.2285584123591193,
                        0.575593004415158,
                        1.4566849270378721,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        3.6649108545747535,
                        0.7852929279154435,
                        1.4398792079001836,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        4.101263296790387,
                        0.9949928514157289,
                        1.4230734887624954,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        4.537615739006021,
                        1.2046927749160143,
                        1.406267769624807,
                    ],
                },
                {
                    "alias": "test_event",
                    "coordinates_absolute": [
                        4.973968181221656,
                        1.4143926984162998,
                        1.3894620504871185,
                    ],
                },
            ],
        },
    ],
)
def test_event_from_dict(input_dict: dict):
    ev = Event.from_dict(input_dict)
    assert isinstance(ev, Event)
    out_dict = ev.to_dict()
    for k, v in out_dict.items():
        assert input_dict[k] == out_dict[k], f"Key {k} not set correctly"

    # Remove a necessary key: should raise an error
    input_dict.pop("alias")
    with pytest.raises(KeyError):
        _ = Event.from_dict(input_dict)


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_AUDIOS[:5])
def test_magic_methods(audio_fpath: str, oyens_space):
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(audio_fpath, "test_event", emitters=emitter)

    # Iterate over all the magic methods that return strings
    for att in ["__str__", "__repr__"]:
        assert isinstance(getattr(ev, att)(), str)

    # Check the __eq__ comparison for identical objects
    assert ev == Event.from_dict(ev.to_dict())

    # Check the __eq__ comparison for non-identical objects
    assert not ev == 123
    assert not ev == "asdf"

    # Removing the emitters and trying to render to dict should give an error
    ev.emitters = None
    with pytest.raises(ValueError):
        _ = ev.to_dict()
