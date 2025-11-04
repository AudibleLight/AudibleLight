#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py"""

import os
from pathlib import Path

import numpy as np
import pytest

from audiblelight import custom_types, utils
from audiblelight.ambience import Ambience
from audiblelight.augmentation import LowpassFilter, Phaser, SpeedUp
from audiblelight.class_mappings import ClassMapping
from audiblelight.core import Scene
from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import (
    Emitter,
    WorldStateRLR,
    WorldStateSOFA,
    get_worldstate_from_string,
)
from tests import utils_tests


@pytest.mark.parametrize(
    "kwargs",
    [
        # Test 1: explicitly define a filepath, emitter keywords, and event keywords (overrides)
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            position=np.array([-0.5, -0.5, 0.5]),
            polar=False,
            duration=5,
            event_start=5,
            scene_start=5,
            augmentations=SpeedUp(stretch_factor=0.5),
        ),
        # Test 2: explicit event keywords and filepath, but no emitter keywords
        dict(filepath=utils_tests.SOUNDEVENT_DIR / "music/001666.mp3", snr=5),
        # Test 3: explicit event and emitter keywords, but no filepath (will be randomly sampled)
        dict(
            position=np.array([-0.5, -0.5, 0.5]),
            polar=False,
            duration=5,
            event_start=5,
            scene_start=5,
            snr=5,
        ),
        # Test 4: no kwargs
        dict(),
        # Test 5: polar position
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "maleSpeech/93853.wav",
            polar=True,
            position=[90, 0.0, 0.5],
            scene_start=0.0,
            augmentations=[LowpassFilter, Phaser()],
        ),
        # Test 6: event right at the end of the scene (but valid)
        dict(
            filepath=utils.sanitise_filepath(utils_tests.TEST_MUSICS[0]),
            scene_start=45,
            duration=4.99,
        ),
        # Test 7: polar position, azimuth exceeds 180 so "wraps around"
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "maleSpeech/93853.wav",
            polar=True,
            # -90 azimuth == straight right
            position=[-90, 0.0, 0.5],
            scene_start=0.0,
        ),
    ],
)
def test_add_event_static(kwargs, oyens_scene_no_overlap: Scene):
    # Clear out the randomly-added mic and add one in a specific position (for reproducibility)
    #  This is necessary because in the final test, we assume a polar position WRT the mic
    #  But if the mic is placed randomly, this polar position can sometimes be invalid
    is_polar = kwargs.get("polar", False)
    if is_polar:
        oyens_scene_no_overlap.clear_microphones()
        oyens_scene_no_overlap.add_microphone(alias="mic000", position=[2.5, -1.0, 1.0])

    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(event_type="static", alias="test_event", **kwargs)
    # Should be added to ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert not ev.is_moving
    assert ev.has_emitters
    assert len(ev) == 1

    # Check that the starting and end time of the event is within temporal bounds for the scene
    assert ev.scene_start >= 0
    assert ev.scene_end < oyens_scene_no_overlap.duration

    # If we've passed in a custom position for the emitter, ensure that this is set correctly
    desired_position = kwargs.get("position", None)

    if desired_position is not None:
        # Non-polar positions, can just check directly
        if not is_polar:
            assert np.array_equal(desired_position, ev.start_coordinates_absolute)

        # Polar positions, a bit more complicated as they're silently converted to polar
        if is_polar:
            cart_pos = utils.polar_to_cartesian(desired_position)[0]
            true_pos = (
                oyens_scene_no_overlap.get_microphone("mic000").coordinates_center
                + cart_pos
            )
            assert np.array_equal(true_pos, ev.start_coordinates_absolute)

            # Also check that we've maintained the correct polar position
            assert np.array_equal(
                desired_position, ev.to_dict()["emitters_relative"]["mic000"][0]
            )

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

    # Check attributes that we will be adding into the event in its __init__ call based on the kwargs
    for attr_ in [
        "event_end",
        "start_coordinates_absolute",
        "end_coordinates_relative_polar",
    ]:
        assert hasattr(ev, attr_)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            duration=5,
            event_start=5,
            scene_start=5,
            shape="semicircular",
            filepath=utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            augmentations=SpeedUp,
        ),
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "music/001666.mp3",
            position=np.array([1.6, -5.1, 1.7]),
            snr=5,
            spatial_velocity=1,
            shape="random",
            duration=5,
        ),
        dict(
            position=np.array([1.6, -5.1, 1.7]),
            duration=5,
            event_start=5,
            scene_start=5,
            snr=5,
            shape="linear",
            spatial_velocity=1,
            spatial_resolution=2,
        ),
        # Test with polar starting positions
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "telephone/30085.wav",
            polar=True,
            position=[0.0, 90.0, 1.0],
            shape="linear",
            scene_start=5.0,  # start five seconds in
            spatial_resolution=1.5,
            spatial_velocity=1.0,
            duration=2,
            augmentations=[Phaser, LowpassFilter],
        ),
        dict(
            filepath=utils_tests.SOUNDEVENT_DIR / "telephone/30085.wav",
            polar=True,
            mic="mic000",
            position=[-90.0, 0.0, 1.0],
            shape="linear",
            scene_start=5.0,
            spatial_resolution=1.5,
            spatial_velocity=1.0,
            duration=2,
            augmentations=[Phaser, LowpassFilter],
        ),
    ],
)
def test_add_moving_event(kwargs, oyens_scene_no_overlap: Scene):
    # Clear out the randomly-added mic and add one in a specific position (for reproducibility)
    #  This is necessary because in the final test, we assume a polar position WRT the mic
    #  But if the mic is placed randomly, this polar position can sometimes be invalid
    is_polar = kwargs.get("polar", False)
    if is_polar:
        oyens_scene_no_overlap.clear_microphones()
        oyens_scene_no_overlap.add_microphone(
            microphone_type="ambeovr", alias="mic000", position=[2.5, -1.0, 1.0]
        )

    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(event_type="moving", alias="test_event", **kwargs)

    # Should have added exactly one event
    assert len(oyens_scene_no_overlap.events) == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert ev.is_moving
    assert ev.has_emitters
    assert len(ev) >= 2

    # Check that the starting and end time of the event is within temporal bounds for the scene
    assert ev.scene_start >= 0
    assert ev.scene_end < oyens_scene_no_overlap.duration

    # If we've passed in a custom position for the emitter, ensure that this is set correctly
    desired_position = kwargs.get("position", None)

    if desired_position is not None:
        # Non-polar positions, can just check directly
        if not is_polar:
            assert np.array_equal(desired_position, ev.start_coordinates_absolute)

        # Polar positions, a bit more complicated as they're silently converted to polar
        if is_polar:
            cart_pos = utils.polar_to_cartesian(desired_position)[0]
            true_pos = (
                oyens_scene_no_overlap.get_microphone("mic000").coordinates_center
                + cart_pos
            )
            assert np.array_equal(true_pos, ev.start_coordinates_absolute)

            # Also check that we've maintained the correct polar position
            assert np.array_equal(
                desired_position, ev.to_dict()["emitters_relative"]["mic000"][0]
            )

    # Should have correct number of sources added to the ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == len(ev)

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

    # Check attributes that we will be adding into the event in its __init__ call based on the kwargs
    for attr_ in [
        "event_end",
        "start_coordinates_absolute",
        "end_coordinates_relative_polar",
    ]:
        assert hasattr(ev, attr_)

    # Starting and ending coordinates should be different
    assert not np.array_equal(
        ev.start_coordinates_absolute, ev.end_coordinates_absolute
    )


@pytest.mark.parametrize(
    "event_kwargs",
    [
        dict(
            filepath=utils_tests.TEST_MUSICS[0],
            duration=5,
            trajectory=None,
            ensure_direct_path=True,
        ),
        dict(
            filepath=utils_tests.TEST_MUSICS[0],
            duration=0.3,
            trajectory=np.array([[2.0, -0.3, 0.5], [2.0, -0.4, 0.5], [2.0, -0.5, 0.5]]),
        ),
    ],
)
def test_add_moving_event_predefined_trajectory(
    event_kwargs, oyens_scene_no_overlap: Scene
):
    # Add microphone to center of the mesh
    oyens_scene_no_overlap.clear_microphones()
    oyens_scene_no_overlap.add_microphone(
        microphone_type="ambeovr", alias="mic000", position=[2.0, -0.5, 1.0]
    )

    # Try N times to add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="predefined", alias="test_event", **event_kwargs
    )

    # Should have added exactly one event
    assert len(oyens_scene_no_overlap.events) == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert ev.is_moving
    assert ev.has_emitters
    assert len(ev) >= 2

    # Should infer characteristics from the trajectory
    assert ev.shape == "predefined"
    assert isinstance(ev.spatial_resolution, custom_types.Numeric)
    assert isinstance(ev.spatial_velocity, custom_types.Numeric)
    assert isinstance(ev.duration, custom_types.Numeric)

    # Expected resolution should be equivalent to the number of emitters over the audio duration, subtracting 1
    expected_resolution = (
        utils.sanitise_positive_number(len(ev) / ev.duration, cast_to=round) - 1
    )
    assert ev.spatial_resolution == expected_resolution

    # Expected velocity is equivalent to the distance travelled in the trajectory over the time
    coords = np.array([em.coordinates_absolute for em in ev.get_emitters()])
    start = coords[0]
    differences = coords[1:] - start
    distances = np.linalg.norm(differences, axis=1)
    max_distance = distances[np.argmax(distances)]
    expected_velocity = max_distance / ev.duration
    assert ev.spatial_velocity == expected_velocity

    # Expected velocity should be within bounds of distribution
    assert ev.spatial_velocity < oyens_scene_no_overlap.event_velocity_dist.max
    assert ev.spatial_velocity > oyens_scene_no_overlap.event_velocity_dist.min


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(shape="sine", ensure_direct_path=True),
        dict(shape="linear", ensure_direct_path=True),
        dict(shape="sawtooth", ensure_direct_path="mic000"),
        dict(shape="semicircular", ensure_direct_path="mic000"),
        dict(shape="random", ensure_direct_path=["mic000"]),
    ],
)
def test_add_event_moving_direct_path(kwargs, oyens_scene_no_overlap: Scene):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        filepath=utils_tests.SOUNDEVENT_DIR / "music/001666.mp3",
        event_type="moving",
        alias="test_event",
        snr=5,
        spatial_velocity=1,
        spatial_resolution=1,
        duration=2,
        **kwargs,
    )

    # Should have added exactly one event
    assert len(oyens_scene_no_overlap.events) == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert ev.is_moving

    # All emitters must have a direct path to the mic
    mic = oyens_scene_no_overlap.get_microphone("mic000")
    for emitter in ev.get_emitters():
        assert oyens_scene_no_overlap.state.path_exists_between_points(
            emitter.coordinates_absolute, mic.coordinates_center
        )


@pytest.mark.parametrize(
    "new_event_audio,event_kwargs,raises",
    [
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=6.0, duration=1.0),
            ValueError,
        ),
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=9.0, duration=10.0),
            ValueError,
        ),
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=3.0, duration=10.0),
            ValueError,
        ),
    ],
)
def test_add_bad_event(
    new_event_audio, event_kwargs, raises, oyens_scene_no_overlap: Scene
):
    """
    Test adding an event that should be rejected
    """
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="static",
        filepath=utils_tests.SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        **event_kwargs,
    )
    # Add the tester event in: should raise an error
    with pytest.raises(raises):
        oyens_scene_no_overlap.add_event(
            event_type="static",
            filepath=new_event_audio,
            alias="bad_event",
            **event_kwargs,
        )
    # Should not be added to the dictionary
    assert len(oyens_scene_no_overlap.events) == 1
    with pytest.raises(KeyError):
        _ = oyens_scene_no_overlap.get_event("bad_event")

    # Test adding an event with an invalid event type
    with pytest.raises(ValueError):
        oyens_scene_no_overlap.add_event(
            event_type="will_fail",
            filepath=new_event_audio,
            alias="bad_event",
            **event_kwargs,
        )


@pytest.mark.parametrize("event_type", ["static", "moving"])
def test_add_event_exceeds_scene_duration(event_type, oyens_scene_no_overlap):
    # This event is too long to be added to the scene, should be rejected
    with pytest.raises(ValueError):
        oyens_scene_no_overlap.add_event(
            event_type=event_type,
            duration=20,
            scene_start=40,
            event_start=0,
            filepath=utils_tests.TEST_MUSICS[0],
        )


@pytest.mark.parametrize(
    "mic_arrays",
    [
        (
            dict(
                microphone_type="ambeovr",
                position=np.array([-0.5, -0.5, 0.5]),
                alias="mic000",
            )
        ),
        (dict(microphone_type="ambeovr", alias="mic000")),
    ],
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


def test_get_event(oyens_scene_no_overlap):
    oyens_scene_no_overlap.add_event(event_type="static")

    # These are all functionally equivalent
    event = oyens_scene_no_overlap.get_event("event000")
    assert isinstance(event, Event)
    event2 = oyens_scene_no_overlap["event000"]
    assert isinstance(event2, Event)
    event3 = oyens_scene_no_overlap.get_event(0)
    assert isinstance(event3, Event)
    event4 = oyens_scene_no_overlap[0]
    assert isinstance(event4, Event)

    # Trying to get event that doesn't exist will raise an error
    #  Error type will vary depending on how we get the event
    with pytest.raises(KeyError):
        oyens_scene_no_overlap.get_event("not_existing")
    with pytest.raises(IndexError):
        oyens_scene_no_overlap.get_event(100)
    with pytest.raises(TypeError):
        oyens_scene_no_overlap.get_event(None)


def test_get_funcs(oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.add_event(event_type="static")
    oyens_scene_no_overlap.add_ambience(alias="tester", noise="white")
    # Test all the get functions
    mic = oyens_scene_no_overlap.get_microphone("mic000")
    assert issubclass(type(mic), MicArray)
    emitter_list = oyens_scene_no_overlap.get_emitters("event000")
    assert isinstance(emitter_list, list)
    emitter = oyens_scene_no_overlap.get_emitter("event000", 0)
    assert isinstance(emitter, Emitter)
    events = oyens_scene_no_overlap.get_events()
    assert isinstance(events, list)
    assert isinstance(events[0], Event)
    ambience = oyens_scene_no_overlap.get_ambience("tester")
    assert isinstance(ambience, Ambience)
    ambiences = oyens_scene_no_overlap.get_ambiences()
    assert isinstance(ambiences, list)
    assert isinstance(ambiences[0], Ambience)
    assert ambiences[0] == ambience


def test_clear_funcs(oyens_scene_no_overlap: Scene):
    # Add an event
    oyens_scene_no_overlap.add_event(event_type="static", alias="remover")
    assert len(oyens_scene_no_overlap.events) == 1
    assert oyens_scene_no_overlap.state.num_emitters == 1
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 1

    # Remove the event and check all has been removed
    oyens_scene_no_overlap.clear_event(alias="remover")
    assert len(oyens_scene_no_overlap.events) == 0
    assert oyens_scene_no_overlap.state.num_emitters == 0
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 0

    # Trying to remove event that doesn't exist will raise an error
    with pytest.raises(KeyError):
        oyens_scene_no_overlap.clear_event("not_existing")

    # By default, the fixture has one mic with four capsules, so check these
    assert len(oyens_scene_no_overlap.state.microphones) == 1
    assert oyens_scene_no_overlap.state.ctx.get_listener_count() == 4

    oyens_scene_no_overlap.clear_emitters()
    oyens_scene_no_overlap.clear_microphones()
    assert len(oyens_scene_no_overlap.state.microphones) == 0
    assert oyens_scene_no_overlap.state.num_emitters == 0

    oyens_scene_no_overlap.add_event(event_type="static", alias="remover")
    oyens_scene_no_overlap.clear_emitter(alias="remover")
    oyens_scene_no_overlap.add_event(event_type="static", alias="remover")
    oyens_scene_no_overlap.clear_emitters()
    assert oyens_scene_no_overlap.state.num_emitters == 0

    oyens_scene_no_overlap.add_microphone(alias="remover")
    oyens_scene_no_overlap.clear_microphone(alias="remover")
    assert len(oyens_scene_no_overlap.state.microphones) == 0


@pytest.mark.parametrize(
    "n_events",
    [
        1,
        2,
        3,
    ],
)
def test_generate_single_mic(n_events: int, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_events()
    for n_event in range(n_events):
        # Use a short duration here so we don't run into issues with placing events
        oyens_scene_no_overlap.add_event(event_type="static", duration=1.0)

    # Suffixes will be stripped out
    oyens_scene_no_overlap.generate(audio_fname="tmp.wav", metadata_fname="tmp.json")

    for fout in ["tmp_mic000.wav", "tmp.json", "tmp_mic000.csv"]:
        assert os.path.isfile(fout)
        os.remove(fout)


@pytest.mark.parametrize(
    "mic1_type,mic2_type",
    [("monocapsule", "ambeovr"), ("eigenmike32", "ambeovr"), ("ambeovr", "ambeovr")],
)
def test_generate_multiple_mics(
    mic1_type: str, mic2_type, oyens_scene_no_overlap: Scene
):
    # Add multiple mics in
    oyens_scene_no_overlap.clear_microphones()
    oyens_scene_no_overlap.add_microphone(
        alias="mic1", microphone_type=mic1_type, keep_existing=True
    )
    oyens_scene_no_overlap.add_microphone(
        alias="mic2", microphone_type=mic2_type, keep_existing=True
    )

    # Add events in
    oyens_scene_no_overlap.clear_events()
    for n_event in range(2):
        # Use a short duration here so we don't run into issues with placing events
        oyens_scene_no_overlap.add_event(event_type="static", duration=1.0)

    # Do the generation
    oyens_scene_no_overlap.generate(
        audio_fname="tmp.wav", metadata_fname="tmp.json", metadata_json=False
    )

    # Check all events
    for event in oyens_scene_no_overlap.get_events():
        # Should have created required attributes for all events
        assert "mic1" in event.spatial_audio.keys()
        assert "mic2" in event.spatial_audio.keys()

        # Audios should be different but have equivalent dims
        mic1_ev = event.spatial_audio["mic1"]
        mic2_ev = event.spatial_audio["mic2"]
        assert mic1_ev.shape[1] == mic2_ev.shape[1]
        assert not np.array_equal(mic1_ev, mic2_ev)

    # Check the scene
    # Should have created required audios in the scene
    assert "mic1" in oyens_scene_no_overlap.audio.keys()
    assert "mic2" in oyens_scene_no_overlap.audio.keys()

    # Audios should be different but have equivalent dims
    mic1 = oyens_scene_no_overlap.audio["mic1"]
    mic2 = oyens_scene_no_overlap.audio["mic2"]
    assert not np.array_equal(mic1, mic2)
    assert mic1.shape[1] == mic2.shape[1]

    # Should have required filepaths
    for fout in ["tmp_mic1.wav", "tmp_mic1.csv", "tmp_mic2.wav", "tmp_mic2.csv"]:
        assert os.path.isfile(fout)
        os.remove(fout)


def test_generated_csv(oyens_scene_no_overlap: Scene):
    # Add a single event in
    oyens_scene_no_overlap.add_event(event_type="static", duration=1.0)

    # Do the generation, save metadata csv file only
    oyens_scene_no_overlap.generate(
        audio=False, metadata_json=False, metadata_fname="tmp"
    )

    # This code is copied from this repo https://github.com/sharathadavanne/seld-dcase2023
    _fid = open("tmp_mic000.csv", "r")
    _output_dict = {}
    for _line in _fid:
        _words = _line.strip().split(",")
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if (
            len(_words) == 5
        ):  # frame, class idx, source_id, polar coordinates(2), no distance data
            _output_dict[_frame_ind].append(
                [int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])]
            )
        if (
            len(_words) == 6
        ):  # frame, class idx, source_id, polar coordinates(2), distance
            _output_dict[_frame_ind].append(
                [int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])]
            )
        elif (
            len(_words) == 7
        ):  # frame, class idx, source_id, cartesian coordinates(3), distance
            _output_dict[_frame_ind].append(
                [
                    int(_words[1]),
                    int(_words[2]),
                    float(_words[3]),
                    float(_words[4]),
                    float(_words[5]),
                ]
            )
    _fid.close()
    # Should have stored some values
    assert len(_output_dict) > 0
    os.remove("tmp_mic000.csv")


@pytest.mark.parametrize(
    "dirpath,raises",
    [(None, False), (os.getcwd(), False), ("not/a/dir", FileNotFoundError)],
)
def test_generate_parse_filepaths(dirpath, raises, oyens_scene_no_overlap):
    oyens_scene_no_overlap.clear_events()
    # Use a short duration here so we don't run into issues with placing events
    oyens_scene_no_overlap.add_event(event_type="static", duration=1.0)

    if not raises:
        oyens_scene_no_overlap.generate(output_dir=dirpath)
        assert len(oyens_scene_no_overlap.audio) > 0
        # Cleanup
        for fout in [
            "audio_out_mic000.wav",
            "metadata_out.json",
            "metadata_out_mic000.csv",
        ]:
            assert os.path.isfile(fout)
            os.remove(fout)
    else:
        with pytest.raises(raises):
            oyens_scene_no_overlap.generate(output_dir=dirpath)
        assert len(oyens_scene_no_overlap.audio) == 0


@pytest.mark.parametrize(
    "noise, filepath",
    [
        ("white", None),
        (2.0, None),
        # Mono audio
        (None, utils_tests.SOUNDEVENT_DIR / "waterTap/95709.wav"),
        # FOA audio
        (
            None,
            utils_tests.TEST_RESOURCES / "spatialsoundevents/voice_whitenoise_foa.wav",
        ),
        # Neither noise nor filepath specified, get random audio from background directory
        (None, None),
    ],
)
def test_add_ambience(noise, filepath, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_ambience()
    oyens_scene_no_overlap.add_ambience(noise=noise, filepath=filepath, alias="tester")
    amb = oyens_scene_no_overlap.get_ambience("tester")
    ambience_audio = amb.load_ambience()
    assert isinstance(ambience_audio, np.ndarray)
    expected_duration = (
        oyens_scene_no_overlap.duration * oyens_scene_no_overlap.sample_rate
    )
    assert ambience_audio.shape == (4, expected_duration)


def test_add_ambience_bad(oyens_scene_no_overlap: Scene):
    # Trying to add ambience with channels not specified and no microphones
    oyens_scene_no_overlap.clear_microphones()
    with pytest.raises(
        ValueError,
        match="Cannot infer Ambience channels when no microphones have been added to the WorldState",
    ):
        oyens_scene_no_overlap.add_ambience()

    # Trying to add ambience with microphones with different number of channels
    oyens_scene_no_overlap.add_microphone(microphone_type="ambeovr")
    oyens_scene_no_overlap.add_microphone(
        microphone_type="eigenmike32", alias="will_break_later"
    )
    with pytest.raises(
        ValueError,
        match="Cannot infer Ambience channels when available microphones have different number of capsules",
    ):
        oyens_scene_no_overlap.add_ambience()
    # Cleanup so we don't get the same error in the next test
    oyens_scene_no_overlap.clear_microphone(alias="will_break_later")

    # Trying to add ambience with duplicate aliases
    oyens_scene_no_overlap.add_ambience(
        filepath=utils_tests.TEST_MUSICS[0], alias="dupe_alias"
    )
    with pytest.raises(KeyError, match="Ambience with alias"):
        oyens_scene_no_overlap.add_ambience(alias="dupe_alias")

    # Trying to add ambience with duplicate audio file when this not permitted
    oyens_scene_no_overlap.allow_duplicate_audios = False
    with pytest.raises(ValueError, match="Audio file"):
        oyens_scene_no_overlap.add_ambience(
            alias="ok", filepath=utils_tests.TEST_MUSICS[0]
        )

    # Reset back to default
    oyens_scene_no_overlap.clear_ambience()
    oyens_scene_no_overlap.clear_microphones()
    oyens_scene_no_overlap.add_microphone(microphone_type="ambeovr")
    oyens_scene_no_overlap.allow_duplicate_audios = True


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "audiblelight_version": "0.1.0",
            "rlr_audio_propagation_version": "0.0.1",
            "creation_time": "2025-08-11_13:07:21",
            "backend": "rlr",
            "duration": 50.0,
            "sample_rate": 44100.0,
            "ref_db": -50,
            "max_overlap": 1,
            "fg_path": [str(utils_tests.SOUNDEVENT_DIR)],
            "bg_path": [str(utils_tests.SOUNDEVENT_DIR)],
            "ambience": {
                "test_ambience": {
                    "alias": "test_ambience",
                    "beta": 1,
                    "filepath": None,
                    "channels": 4,
                    "sample_rate": 44100.0,
                    "duration": 10.0,
                    "ref_db": -50,
                    "noise_kwargs": {},
                }
            },
            "events": {
                "test_event": {
                    "alias": "test_event",
                    "filename": "000010.mp3",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "music/000010.mp3"),
                    "class_id": 8,
                    "class_label": "music",
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
                    "shape": "static",
                    "num_emitters": 1,
                    "emitters": [
                        [1.8156068957785347, -1.863507837016133, 1.8473540916136413]
                    ],
                    "emitters_relative": {
                        "mic000": [
                            [-156.0890612441748, -5.976352087676762, 3.3744825372046803]
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
                        }
                    ],
                    "ref_ir_channel": 0,
                    "direct_path_time_ms": [6, 30],
                }
            },
            "state": {
                "backend": "rlr",
                "sample_rate": 44100.0,
                "emitters": {
                    "test_event": [
                        [1.8156068957785347, -1.863507837016133, 1.8473540916136413]
                    ]
                },
                "microphones": {
                    "mic000": {
                        "name": "ambeovr",
                        "micarray_type": "AmbeoVR",
                        "is_spherical": True,
                        "channel_layout_type": "mic",
                        "n_capsules": 4,
                        "capsule_names": ["FLU", "FRD", "BLD", "BRU"],
                        "coordinates_absolute": [
                            [2.426383418709459, -4.797424158428278, 0.6561976663024979],
                            [2.426383418709459, -4.80900871773507, 0.644726137575477],
                            [2.414798859402668, -4.797424158428278, 0.644726137575477],
                            [2.414798859402668, -4.80900871773507, 0.6561976663024979],
                        ],
                        "coordinates_center": [
                            2.4205911390560635,
                            -4.803216438081674,
                            0.6504619019389875,
                        ],
                    }
                },
                "mesh": {
                    "fname": "Oyens",
                    "ftype": ".glb",
                    "fpath": str(utils_tests.OYENS_PATH),
                    "units": "meters",
                    "from_gltf_primitive": False,
                    "name": "defaultobject",
                    "node": "defaultobject",
                    "bounds": [
                        [-3.0433080196380615, -10.448445320129395, -1.1850370168685913],
                        [5.973234176635742, 2.101027011871338, 2.4577369689941406],
                    ],
                    "centroid": [
                        1.527919030159762,
                        -4.550817438070386,
                        1.162934397641578,
                    ],
                },
                "rlr_config": {
                    "diffraction": 1,
                    "direct": 1,
                    "direct_ray_count": 500,
                    "direct_sh_order": 3,
                    "frequency_bands": 4,
                    "global_volume": 1.0,
                    "hrtf_back": [0.0, 0.0, 1.0],
                    "hrtf_right": [1.0, 0.0, 0.0],
                    "hrtf_up": [0.0, 1.0, 0.0],
                    "indirect": 1,
                    "indirect_ray_count": 5000,
                    "indirect_ray_depth": 200,
                    "indirect_sh_order": 1,
                    "max_diffraction_order": 10,
                    "max_ir_length": 4.0,
                    "mesh_simplification": 0,
                    "sample_rate": 44100.0,
                    "size": 146,
                    "source_ray_count": 200,
                    "source_ray_depth": 10,
                    "temporal_coherence": 0,
                    "thread_count": 1,
                    "transmission": 1,
                    "unit_scale": 1.0,
                },
                "empty_space_around_mic": 0.1,
                "empty_space_around_emitter": 0.2,
                "empty_space_around_surface": 0.2,
                "empty_space_around_capsule": 0.05,
                "repair_threshold": None,
            },
            "class_mapping": {
                "femaleSpeech": 0,
                "maleSpeech": 1,
                "clapping": 2,
                "telephone": 3,
                "laughter": 4,
                "domesticSounds": 5,
                "footsteps": 6,
                "doorCupboard": 7,
                "music": 8,
                "musicInstrument": 9,
                "waterTap": 10,
                "bell": 11,
                "knock": 12,
            },
        },
        {
            "audiblelight_version": "0.1.0",
            "rlr_audio_propagation_version": "0.0.1",
            "creation_time": "2025-10-30_11:18:02",
            "duration": 50.0,
            "backend": "SOFA",
            "sample_rate": 44100,
            "ref_db": -65,
            "max_overlap": 1,
            "fg_path": [str(utils_tests.SOUNDEVENT_DIR)],
            "bg_path": [str(utils_tests.BACKGROUND_DIR)],
            "ambience": {
                "ambience000": {
                    "alias": "ambience000",
                    "beta": 0,
                    "filepath": None,
                    "channels": 4,
                    "sample_rate": 44100,
                    "duration": 50.0,
                    "ref_db": -65,
                    "noise_kwargs": {},
                }
            },
            "events": {
                "event000": {
                    "alias": "event000",
                    "filename": "70345.wav",
                    "filepath": str(
                        utils_tests.SOUNDEVENT_DIR / "doorCupboard/70345.wav"
                    ),
                    "class_id": 7,
                    "class_label": "doorCupboard",
                    "is_moving": False,
                    "scene_start": 11.008224860918492,
                    "scene_end": 12.320447083140714,
                    "event_start": 0.0,
                    "event_end": 1.3122222222222222,
                    "duration": 1.3122222222222222,
                    "snr": 25.595045335730944,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "shape": "static",
                    "num_emitters": 1,
                    "emitters": [[-1.5, -1.5, 1.0]],
                    "emitters_relative": {
                        "mic000": [[-135.0, 25.23940182067891, 2.345207879911715]]
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
                        }
                    ],
                    "ref_ir_channel": 0,
                    "direct_path_time_ms": [6, 30],
                }
            },
            "state": {
                "backend": "SOFA",
                "sofa": str(utils_tests.METU_SOFA_PATH),
                "sample_rate": 44100,
                "emitters": {"event000": [[-1.5, -1.5, 1.0]]},
                "emitter_sofa_idxs": {"event000": [132]},
                "microphones": {
                    "mic000": {
                        "name": "em32",
                        "micarray_type": "_DynamicMicArray",
                        "is_spherical": False,
                        "channel_layout_type": "foa",
                        "n_capsules": 4,
                        "capsule_names": ["1", "2", "3", "4"],
                        "coordinates_absolute": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        "coordinates_center": [0.0, 0.0, 0.0],
                        "coordinates_polar": None,
                        "coordinates_cartesian": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                    }
                },
                "metadata": {
                    "bounds": [[-1.5, -1.5, -1.0], [1.5, 1.5, 1.0]],
                    "Conventions": "SOFA",
                    "Version": "2.1",
                    "SOFAConventions": "SingleRoomSRIR",
                    "SOFAConventionsVersion": "1.0",
                    "APIName": "pysofaconventions",
                    "APIVersion": "0.1.5",
                    "AuthorContact": "chris.ick@nyu.edu",
                    "Organization": "Music and Audio Research Lab - NYU",
                    "License": "Use whatever you want",
                    "DataType": "FIR",
                    "DateCreated": "Thu Apr 11 19:39:03 2024",
                    "DateModified": "Thu Apr 11 19:39:03 2024",
                    "Title": "METU-SPARG - classroom",
                    "RoomType": "shoebox",
                    "DatabaseName": "METU-SPARG",
                    "ListenerShortName": "em32",
                    "RoomShortName": "classroom",
                    "Comment": "N/A",
                },
            },
            "class_mapping": {
                "femaleSpeech": 0,
                "maleSpeech": 1,
                "clapping": 2,
                "telephone": 3,
                "laughter": 4,
                "domesticSounds": 5,
                "footsteps": 6,
                "doorCupboard": 7,
                "music": 8,
                "musicInstrument": 9,
                "waterTap": 10,
                "bell": 11,
                "knock": 12,
            },
        },
    ],
)
def test_scene_from_dict(input_dict: dict):
    ev = Scene.from_dict(input_dict)
    assert isinstance(ev, Scene)

    # Check number of events, ambiences, emitters, microphones
    assert len(ev.events) == len(input_dict["events"])
    assert len(ev.ambience.keys()) == len(input_dict["ambience"])
    assert ev.state.num_emitters == sum(
        len(em) for em in input_dict["state"]["emitters"].values()
    )
    assert len(ev.state.microphones) == len(input_dict["state"]["microphones"])

    # Check serialising back and forth to dictionary
    out_dict = ev.to_dict()
    assert Scene.from_dict(out_dict) == ev

    # Check source count in ray-tracing engine
    if ev.state.name == "RLR":
        assert ev.state.num_emitters == ev.state.ctx.get_source_count()


@pytest.mark.parametrize(
    "filepath, kwargs",
    [
        # Test 1: explicitly define a filepath, emitter keywords, and event keywords (overrides)
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(
                position=np.array([-0.5, -0.5, 0.5]),
                polar=False,
                ensure_direct_path=False,
                duration=5,
                event_start=5,
                scene_start=5,
            ),
        ),
    ],
)
def test_magic_methods(filepath, kwargs, oyens_scene_no_overlap):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="static", filepath=filepath, alias="test_event", **kwargs
    )
    # Creating an iterator
    for event in oyens_scene_no_overlap:
        assert isinstance(event, Event)
    # Testing remaining dunders
    for att in ["__len__", "__str__", "__repr__", "__getitem__"]:
        assert hasattr(oyens_scene_no_overlap, att)
        _ = getattr(oyens_scene_no_overlap, att)
    # Dump the scene again, reload, and compare
    ev2 = Scene.from_dict(oyens_scene_no_overlap.to_dict())
    assert ev2 == oyens_scene_no_overlap


@pytest.mark.parametrize(
    "aug_list, n_augs",
    [
        ([Phaser, LowpassFilter, SpeedUp], 2),
        ([Phaser], 1),
        # requesting more augs than we have available, coerced to 2
        ([Phaser, LowpassFilter], 5),
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        dict(event_type="static"),
        dict(
            event_type="moving",
            spatial_resolution=1,
            spatial_velocity=1,
            shape="linear",
        ),
    ],
)
def test_add_events_with_random_augmentations(aug_list, n_augs, params):
    """
    Add events with N random augmentations, drawn from our list
    """
    sc = Scene(
        duration=50,
        backend="rlr",
        sample_rate=44100,
        event_augmentations=aug_list,
        fg_path=utils_tests.SOUNDEVENT_DIR,
        max_overlap=1,
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
    )
    ev = sc.add_event(
        augmentations=n_augs,
        filepath=utils_tests.SOUNDEVENT_DIR / "telephone/30085.wav",
        duration=1,
        **params,
    )
    augs = ev.get_augmentations()

    # All augs should be sampled from our list
    for aug in augs:
        assert type(aug) in aug_list

    # Augmentations should always be unique when random sampling
    assert len(augs) == len(set(type(aug) for aug in augs))

    # Should have expected number of augs
    assert len(augs) == min(len(sc.event_augmentations), n_augs)


@pytest.mark.parametrize(
    "aug_list_of_tuples, n_augs",
    [
        (
            [
                (Phaser, dict(mix=np.random.rand)),
                (LowpassFilter, dict(cutoff_frequency_hz=5000)),
            ],
            2,
        ),
        (
            [
                (SpeedUp, dict(stretch_factor=np.random.rand)),
                (LowpassFilter, dict(cutoff_frequency_hz=5000)),
            ],
            1,
        ),
        (
            [
                (SpeedUp, dict(stretch_factor=np.random.rand)),
                (
                    Phaser,
                    dict(
                        mix=np.random.rand,
                        feedback=0.5,
                        depth=np.random.rand,
                        centre_frequency_hz=500,
                    ),
                ),
                (LowpassFilter, dict(cutoff_frequency_hz=5000)),
            ],
            3,
        ),
    ],
)
def test_add_events_with_parametrised_augmentations(aug_list_of_tuples, n_augs):
    """
    Test adding events where we've set a parametrised distribution for each augmentation
    """
    sc = Scene(
        duration=50,
        sample_rate=44100,
        event_augmentations=aug_list_of_tuples,
        fg_path=utils_tests.SOUNDEVENT_DIR,
        max_overlap=1,
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
    )
    ev = sc.add_event(augmentations=n_augs, event_type="static")
    augs = ev.get_augmentations()

    # Sort everything so the FX are in the correct order
    augs = sorted(augs, key=lambda x: x.name)
    aug_list_of_tuples = [
        i
        for i in sorted(aug_list_of_tuples, key=lambda x: x[0]().name)
        if i[0] in (type(a) for a in augs)
    ]

    # Should have the correct number of augmentations
    assert len(augs) == n_augs

    # Iterate over the passed in list of augs and the actual augs
    for (expected_aug_type, expected_aug_kwargs), actual_aug in zip(
        aug_list_of_tuples, augs
    ):
        # Should be the correct type
        assert isinstance(actual_aug, expected_aug_type)

        # Iterate over all the kwargs we were expecting to use
        for k, v in expected_aug_kwargs.items():

            # Explicitly provided a value, should be used
            if isinstance(v, custom_types.Numeric):
                assert getattr(actual_aug, k) == v

            # Provided a function to sample the value, should be within range
            else:
                # Call the function N times and check that the actual value is within the range
                sampled_values = [v() for _ in range(1000)]
                assert (
                    (min(sampled_values) - utils.SMALL)
                    < getattr(actual_aug, k)
                    < (max(sampled_values) + utils.SMALL)
                )


@pytest.mark.parametrize(
    "fg_path,bg_path",
    [
        # Single paths
        (
            utils_tests.SOUNDEVENT_DIR / "music",
            str(utils_tests.SOUNDEVENT_DIR / "waterTap"),
        ),
        # list of paths
        (
            [
                utils_tests.SOUNDEVENT_DIR / "music",
                utils_tests.SOUNDEVENT_DIR / "femaleSpeech",
            ],
            [
                utils_tests.SOUNDEVENT_DIR / "waterTap",
                str(utils_tests.SOUNDEVENT_DIR / "maleSpeech"),
            ],
        ),
        # No paths
        (None, None),
    ],
)
def test_parse_audio_paths(fg_path, bg_path):
    sc = Scene(
        duration=50,
        fg_path=fg_path,
        bg_path=bg_path,
        sample_rate=44100,
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
    )
    assert isinstance(sc.fg_audios, list)
    assert isinstance(sc.bg_audios, list)

    for path, audios in zip([fg_path, bg_path], [sc.fg_audios, sc.bg_audios]):
        if path is not None:
            assert len(audios) > 0
            out = sc._get_random_audio(audios)
            assert isinstance(out, Path)
            assert out.is_file()
            assert out.suffix.replace(".", "") in custom_types.AUDIO_EXTS
        else:
            assert len(audios) == 0
            with pytest.raises(FileNotFoundError):
                _ = sc._get_random_audio(audios)


@pytest.mark.parametrize("bad_event_type", ["static", "moving"])
def test_add_duplicated_event_audio(bad_event_type):
    sc = Scene(
        duration=50,
        allow_duplicate_audios=False,
        sample_rate=44100,
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
    )

    # Add the audio in the first time: should be fine
    ok_event_type = "static" if bad_event_type == "moving" else "moving"
    sc.add_event(
        event_type=ok_event_type,
        alias="ok",
        filepath=utils_tests.TEST_MUSICS[0],
        duration=1.0,
    )

    # Add the audio in the second time: should raise an error
    with pytest.raises(ValueError, match="Audio file"):
        sc.add_event(
            event_type=bad_event_type,
            alias="bad",
            filepath=utils_tests.TEST_MUSICS[0],
            duration=1.0,
        )


def test_add_duplicated_ambience_audio():
    sc = Scene(
        duration=50,
        allow_duplicate_audios=False,
        sample_rate=44100,
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
    )

    # Add the audio in the first time should be fine
    sc.add_ambience(alias="ok", filepath=utils_tests.TEST_MUSICS[0], channels=4)

    # Add the ambience in the second time, should raise
    with pytest.raises(ValueError, match="Audio file"):
        sc.add_ambience(alias="bad", filepath=utils_tests.TEST_MUSICS[0], channels=4)


@pytest.mark.parametrize("allow_dupes", [True, False])
def test_get_random_audio_dupes(allow_dupes):
    sc = Scene(
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
        duration=50,
        allow_duplicate_audios=allow_dupes,
        sample_rate=44100,
    )

    chosen_audio = utils.sanitise_filepath(utils_tests.TEST_MUSICS[0])
    sc.add_event(duration=1, event_type="static", filepath=chosen_audio)

    # Get N random audios
    randoms = [sc._get_random_audio(utils_tests.TEST_MUSICS) for _ in range(50)]

    # Allow for duplicates in the result or not depending on the argument
    if allow_dupes:
        assert chosen_audio in randoms
    else:
        assert chosen_audio not in randoms


@pytest.mark.parametrize("allow_same_class", [True, False])
def test_get_random_audio_no_same_class_events(allow_same_class):
    sc = Scene(
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
        duration=50,
        allow_duplicate_audios=True,
        allow_same_class_events=allow_same_class,
        sample_rate=44100,
        fg_path=utils_tests.SOUNDEVENT_DIR,
        class_mapping="dcase2023task3",
    )

    # Add a set audio file with music class
    chosen_audio = utils.sanitise_filepath(utils_tests.TEST_MUSICS[0])
    sc.add_event(duration=1, event_type="static", filepath=chosen_audio)

    # Get more audio files
    randoms = [sc._get_random_audio() for _ in range(50)]

    # Map the chosen audio files to labels with the scene's mapping object
    mapped = set(
        [sc.class_mapping.infer_label_idx_from_filepath(ap)[1] for ap in randoms]
    )

    # If not allowing same classes, we shouldn't get any music audio files
    if not allow_same_class:
        assert "music" not in mapped
    else:
        assert "music" in mapped


@pytest.mark.parametrize(
    "audio1, audio2, raises",
    [
        ("waterTap/95709.wav", "waterTap/205695.wav", True),
        ("music/000010.mp3", "music/001666.mp3", True),
        ("music/000010.mp3", "musicInstrument/3471.wav", False),
        ("maleSpeech/93853.wav", "femaleSpeech/236385.wav", False),
    ],
)
def test_add_duplicated_class_event(audio1, audio2, raises):
    sc = Scene(
        backend="rlr",
        backend_kwargs=dict(
            mesh=utils_tests.OYENS_PATH,
        ),
        duration=50,
        allow_same_class_events=False,
        sample_rate=44100,
        class_mapping="dcase2023task3",
    )

    # Add first audio path, should be OK
    sc.add_event(event_type="static", filepath=utils_tests.SOUNDEVENT_DIR / audio1)

    # Trying to add second audio path should raise an error
    if raises:
        with pytest.raises(ValueError):
            sc.add_event(
                event_type="static", filepath=utils_tests.SOUNDEVENT_DIR / audio2
            )
    else:
        sc.add_event(event_type="static", filepath=utils_tests.SOUNDEVENT_DIR / audio2)
        assert len(sc.get_events()) == 2


@pytest.mark.parametrize("n_events", [1, 2, 3])
def test_generate_foa(n_events: int, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.clear_microphones()

    # Add FOA capsule microphone + events then do generation
    oyens_scene_no_overlap.add_microphone(
        microphone_type="foalistener", alias="foa_test", keep_existing=False
    )
    for n in range(n_events):
        oyens_scene_no_overlap.add_event(event_type="static", duration=1)
    oyens_scene_no_overlap.generate(
        audio=False, metadata_json=False, metadata_dcase=False
    )

    # Shape of the audio should be 4 channel
    assert oyens_scene_no_overlap.audio["foa_test"].shape[0] == 4
    for ev in oyens_scene_no_overlap.events.values():
        assert ev.spatial_audio["foa_test"].shape[0] == 4
        assert ev.spatial_audio["foa_test"].shape[1] > 0
        assert not np.all(ev.spatial_audio["foa_test"] == 0)

    # Test the IRs
    foa = oyens_scene_no_overlap.get_microphone("foa_test")
    assert foa.channel_layout_type == "foa"

    # Should have expected shape
    n_caps, n_emits, n_samps = foa.irs.shape
    assert n_caps == 4
    assert n_emits == sum(len(i) for i in oyens_scene_no_overlap.get_events())
    assert n_samps >= 1

    # Reshape so that all we get channels * emitters, samples
    res = foa.irs.reshape(n_caps * n_emits, n_samps)

    # Check that no channel is all zeroes
    assert not np.any(np.all(res == 0, axis=1))

    # Check that no channel is a copy of another: all must be unique
    assert len(np.unique(res, axis=0)) == len(res)

    # Iterate over all emitters
    for emitter_idx in range(n_emits):
        irs_at_emitter = foa.irs[:, emitter_idx, :]

        # Check that the IRs are all "from" this emitter
        #  We can do this by checking that the first non-zero value
        #  occurs at roughly the same time for every channel
        mask: np.ndarray = irs_at_emitter != 0
        ereflect = mask.argmax(axis=1)
        assert np.max(ereflect) - np.min(ereflect) < 10


@pytest.mark.parametrize(
    "overrides",
    [
        dict(
            event_type="static",
            scene_start=10,
            duration=5,
            event_start=1,
        ),
        dict(event_type="static", snr=5),
        dict(
            event_type="moving",
            scene_start=10,
            duration=5,
            event_start=1,
            spatial_resolution=2,
            spatial_velocity=2,
        ),
    ],
)
def test_add_event_overrides(overrides, oyens_scene_no_overlap: Scene):
    # Use a music file as this should be nice and long
    oyens_scene_no_overlap.clear_events()
    created = oyens_scene_no_overlap.add_event(
        filepath=utils_tests.TEST_MUSICS[0], **overrides
    )

    # Define the kwargs we'll use for all event types
    all_kwargs = ["scene_start", "event_start", "duration", "snr"]
    if created.is_moving:
        all_kwargs.extend(["spatial_resolution", "spatial_velocity"])

    # Iterate over all expected kwargs
    for kw in all_kwargs:

        # Event must have the kwarg
        assert hasattr(created, kw)

        # If set as an override, should be used directly
        if kw in overrides:
            assert getattr(created, kw) == overrides[kw]

        # Otherwise, should be in range of min/max values of dist
        else:
            dist = getattr(oyens_scene_no_overlap, kw + "_dist", None)
            if dist is not None:
                assert dist.min <= getattr(created, kw) <= dist.max


@pytest.mark.parametrize(
    "backend,kwargs",
    [
        # Test with strings
        ("rlr", dict(mesh=utils_tests.OYENS_PATH)),
        ("sofa", dict(sofa=utils_tests.METU_SOFA_PATH)),
        # Test with initialised backends
        (
            WorldStateSOFA,
            dict(
                sofa=utils_tests.METU_SOFA_PATH,
                sample_rate=44100,
            ),
        ),
        (
            WorldStateRLR,
            dict(
                mesh=utils_tests.OYENS_PATH,
                sample_rate=44100,
                add_to_context=True,  # update worldstate with every addition
                empty_space_around_emitter=0.2,  # all in meters
                empty_space_around_mic=0.1,  # all in meters
                empty_space_around_surface=0.2,  # all in meters
                waypoints_json=utils_tests.OYENS_WAYPOINTS_PATH,
            ),
        ),
    ],
)
def test_parse_backend(backend, kwargs):
    if not isinstance(backend, str):
        backend = backend(**kwargs)
        sc = Scene(
            duration=60,
            backend=backend,
        )
    else:
        sc = Scene(duration=60, backend=backend, backend_kwargs=kwargs)

    if isinstance(backend, str):
        expected_ws = get_worldstate_from_string(backend)
    else:
        expected_ws = type(backend)

    # Type should be identical
    assert type(sc.state) is expected_ws
    assert getattr(sc.state, "sample_rate") == 44100

    if expected_ws is WorldStateRLR:
        assert str(sc.state.mesh.metadata["fpath"]) == str(utils_tests.OYENS_PATH)
    else:
        assert str(getattr(sc.state, "sofa_path")) == str(utils_tests.METU_SOFA_PATH)


@pytest.mark.parametrize(
    "backend,expected",
    [
        (12345, TypeError),
        (
            WorldStateRLR(
                mesh=utils_tests.OYENS_PATH,
                sample_rate=123456,
                add_to_context=True,  # update worldstate with every addition
                empty_space_around_emitter=0.2,  # all in meters
                empty_space_around_mic=0.1,  # all in meters
                empty_space_around_surface=0.2,  # all in meters
                waypoints_json=utils_tests.OYENS_WAYPOINTS_PATH,
            ),
            ValueError,
        ),
    ],
)
def test_parse_backend_failure(backend, expected):
    with pytest.raises(expected):
        _ = Scene(
            duration=60,
            sample_rate=44100,
            backend=backend,
        )


@pytest.mark.parametrize(
    "filepath,mapping,expected",
    [
        (utils_tests.TEST_MUSICS[0], "dcase2023task3", 8),
        (
            utils_tests.TEST_RESOURCES / "soundevents/waterTap/95709.wav",
            dict(
                music=1,
                musicInstrument=2,
                waterTap=3,
                anotherClass=4,
                anotherClassAgain=5,
            ),
            3,
        ),
    ],
)
@pytest.mark.parametrize("event_type", ["static", "moving"])
def test_parse_class_mapping(filepath, mapping, expected, event_type):
    sc = Scene(
        backend="rlr",
        duration=50,
        sample_rate=22050,
        class_mapping=mapping,
        backend_kwargs=dict(mesh=utils_tests.OYENS_PATH),
    )

    # Add in the event
    sc.add_event(
        event_type=event_type, filepath=filepath, alias="class_mapping", duration=1
    )

    # Grab the event
    ev = sc.get_event("class_mapping")

    # ID should have been passed correctly
    assert ev.class_id == expected

    current_mapping = sc.get_class_mapping()
    assert ev.class_label in current_mapping
    assert expected == current_mapping[ev.class_label]

    # Should be a ClassMapping child
    assert issubclass(type(sc.class_mapping), ClassMapping)
