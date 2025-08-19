#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/event.py"""

from typing import Optional

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.augmentation import (
    Augmentation,
    Compressor,
    LowpassFilter,
    MultibandEqualizer,
    Phaser,
    PitchShift,
    TimeShift,
)
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
    for k in ["alias", "duration", "filepath", "emitters", "emitters_relative"]:
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
            "scene_end": 0.3922902494331066,
            "event_start": 0.0,
            "event_end": 0.3922902494331066,
            "duration": 0.3922902494331066,
            "snr": None,
            "sample_rate": 44100.0,
            "spatial_resolution": None,
            "spatial_velocity": None,
            "num_emitters": 1,
            "emitters": [[1.8156068957785347, -1.863507837016133, 1.8473540916136413]],
            "emitters_relative": {
                "mic000": [[203.9109387558252, -5.976352087676762, 3.3744825372046803]]
            },
            "augmentations": [],
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
            "snr": 5.715503046168063,
            "sample_rate": 44100.0,
            "spatial_resolution": 3.544083664851323,
            "spatial_velocity": 0.5979377187987713,
            "num_emitters": 19,
            "emitters": [
                [2.7111620345263807, -1.801159962586631, 1.1798789132922645],
                [2.7193423471675033, -1.8972692055682074, 1.2374976026869187],
                [2.7275226598086255, -1.9933784485497836, 1.2951162920815729],
                [2.735702972449748, -2.08948769153136, 1.352734981476227],
                [2.7438832850908703, -2.1855969345129362, 1.4103536708708813],
                [2.752063597731993, -2.2817061774945127, 1.4679723602655357],
                [2.760243910373115, -2.377815420476089, 1.5255910496601899],
                [2.7684242230142377, -2.473924663457665, 1.583209739054844],
                [2.77660453565536, -2.5700339064392415, 1.6408284284494983],
                [2.7847848482964825, -2.666143149420818, 1.6984471178441525],
                [2.792965160937605, -2.7622523924023943, 1.7560658072388067],
                [2.8011454735787273, -2.8583616353839707, 1.8136844966334609],
                [2.80932578621985, -2.954470878365547, 1.8713031860281153],
                [2.817506098860972, -3.050580121347123, 1.9289218754227693],
                [2.8256864115020948, -3.1466893643286995, 1.9865405648174237],
                [2.833866724143217, -3.2427986073102755, 2.0441592542120777],
                [2.8420470367843396, -3.338907850291852, 2.101777943606732],
                [2.8502273494254617, -3.4350170932734283, 2.1593966330013865],
                [2.8584076620665844, -3.5311263362550047, 2.2170153223960405],
            ],
            "emitters_relative": {
                "mic000": [
                    [65.1882226511131, -2.057845599201502, 3.9287039209690207],
                    [64.47613615162182, -1.2441553279867532, 3.8435683917328642],
                    [63.73253315145274, -0.393717606119576, 3.759862909755683],
                    [62.95554187499614, 0.49515327065349185, 3.6776851215571598],
                    [62.14316789977479, 1.4241399377299937, 3.5971397305953015],
                    [61.2932880430769, 2.3948983690896375, 3.518338850149241],
                    [60.403644708907294, 3.4090286444040867, 3.441402319178861],
                    [59.47184097418077, 4.468039280843969, 3.3664579628101285],
                    [58.49533677037989, 5.573304353672314, 3.2936417751819493],
                    [57.47144661034538, 6.7260127099415215, 3.223097998202599],
                    [56.39733942140763, 7.927108744178863, 3.1549790655009753],
                    [55.270041177193676, 9.177224480142927, 3.0894453768304633],
                    [54.086441171722555, 10.476603116017898, 3.0266648648350145],
                    [52.84330294978376, 11.825014766920178, 2.966812314011842],
                    [51.537281093449685, 13.221665895734276, 2.9100683916457624],
                    [50.1649452583885, 14.665104863290383, 2.8566183533116773],
                    [48.722813042342466, 16.15312712955325, 2.8066503921643675],
                    [47.207393431065505, 17.682684842297828, 2.7603536125166257],
                    [45.61524267370456, 19.24980675934964, 2.7179156247699607],
                ]
            },
            "augmentations": [
                {
                    "name": "Phaser",
                    "sample_rate": 44100,
                    "rate_hz": 9.480337646552867,
                    "depth": 0.4725113710968438,
                    "centre_frequency_hz": 2348.1728842622597,
                    "feedback": 0.0810976870856293,
                    "mix": 0.4228090059318278,
                },
                {
                    "name": "LowpassFilter",
                    "sample_rate": 44100,
                    "cutoff_frequency_hz": 7561.425049192692,
                },
            ],
        },
    ],
)
def test_event_from_dict(input_dict: dict):
    ev = Event.from_dict(input_dict)
    assert isinstance(ev, Event)

    # Check augmentations initialised correctly
    if len(input_dict["augmentations"]) > 0:
        for augmentation in ev.get_augmentations():
            assert issubclass(type(augmentation), Augmentation)

    out_dict = ev.to_dict()
    for k, v in out_dict.items():
        if not isinstance(input_dict[k], list):
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
    # Add some augmentations
    ev.register_augmentations(TimeShift)
    ev.register_augmentations([PitchShift, LowpassFilter])

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


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_AUDIOS[:5])
# Define some example augmentation chains
@pytest.mark.parametrize(
    "augmentations",
    [
        (LowpassFilter, MultibandEqualizer, Compressor),
        (Phaser, Compressor),
        (PitchShift, TimeShift),
    ],
)
def test_add_augmentations(audio_fpath, augmentations):
    ev = Event(
        audio_fpath,
        "test_event",
    )
    # Load up the pre-augmented audio
    init_audio = ev.load_audio(ignore_cache=True)

    # Add in the augmentations
    ev.register_augmentations(augmentations)
    # Should have invalidated the audio we cached
    assert ev.audio is None
    assert len(ev.augmentations) > 0

    # Audio should be different to the initial form after augmentation
    aug_audio = ev.load_audio()
    assert not np.array_equal(init_audio, aug_audio)

    # However, audio should have the same shape after augmentation
    try:
        utils.validate_shape(aug_audio.shape, init_audio.shape)
    except ValueError as e:
        pytest.fail(reason=e)
