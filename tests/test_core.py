#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py"""

import os
from pathlib import Path

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import Emitter
from tests import utils_tests


@pytest.mark.parametrize(
    "filepath,emitter_kws,event_kws",
    [
        # Test 1: explicitly define a filepath, emitter keywords, and event keywords (overrides)
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(
                position=np.array([-0.5, -0.5, 0.5]),
                polar=False,
                ensure_direct_path=False,
            ),
            dict(duration=5, event_start=5, scene_start=5),
        ),
        # Test 2: explicit event keywords and filepath, but no emitter keywords
        (
            utils_tests.SOUNDEVENT_DIR / "music/001666.mp3",
            None,
            dict(snr=5, spatial_velocity=5),
        ),
        # Test 3: explicit event and emitter keywords, but no filepath (will be randomly sampled)
        (
            None,
            dict(
                position=np.array([-0.5, -0.5, 0.5]),
                polar=False,
                ensure_direct_path=False,
            ),
            dict(duration=5, event_start=5, scene_start=5, snr=5, spatial_velocity=5),
        ),
        # Test 4: no path, no kwargs
        (None, None, None),
    ],
)
def test_add_event_static(
    filepath: str, emitter_kws, event_kws, oyens_scene_no_overlap: Scene
):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="static",
        filepath=filepath,
        alias="test_event",
        emitter_kwargs=emitter_kws,
        event_kwargs=event_kws,
    )
    # Should be added to ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert not ev.is_moving
    assert ev.has_emitters
    assert len(ev.emitters) == 1

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
    for attr_ in [
        "event_end",
        "start_coordinates_absolute",
        "end_coordinates_relative_polar",
    ]:
        assert hasattr(ev, attr_)


@pytest.mark.parametrize(
    "filepath,emitter_kws,event_kws",
    [
        # Predefine a starting position for speedups
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(),
            dict(duration=5, event_start=5, scene_start=5),
        ),
        (
            utils_tests.SOUNDEVENT_DIR / "music/001666.mp3",
            dict(starting_position=np.array([1.6, -5.1, 1.7])),
            dict(snr=5, spatial_velocity=1, duration=5),
        ),
        (
            None,
            dict(starting_position=np.array([1.6, -5.1, 1.7])),
            dict(
                duration=5,
                event_start=5,
                scene_start=5,
                snr=5,
                spatial_velocity=1,
                spatial_resolution=2,
            ),
        ),
    ],
)
def test_add_moving_event(
    filepath: str, emitter_kws, event_kws, oyens_scene_no_overlap: Scene
):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="moving",
        filepath=filepath,
        alias="test_event",
        emitter_kwargs=emitter_kws,
        event_kwargs=event_kws,
    )

    # Should have added exactly one event
    assert len(oyens_scene_no_overlap.events) == 1

    # Get the event
    ev = oyens_scene_no_overlap.get_event("test_event")
    assert isinstance(ev, Event)
    assert ev.is_moving
    assert ev.has_emitters
    assert len(ev.emitters) >= 2

    # Should have correct number of sources added to the ray-tracing engine
    assert oyens_scene_no_overlap.state.ctx.get_source_count() == len(ev.emitters)

    # Check all overrides passed correctly to the event class
    #  When we're using a random file, we cannot check these variables as they might have changed
    #  due to cases where the duration of the random file is shorter than the passed value (5 seconds)
    if filepath is not None and event_kws is not None:
        for override_key, override_val in event_kws.items():
            assert getattr(ev, override_key) == override_val

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
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(sample_rate=12345),
            ValueError,
        ),  # sample rate different to state
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
        event_kwargs=dict(scene_start=5.0, duration=5.0),
    )
    # Add the tester event in: should raise an error
    with pytest.raises(raises):
        oyens_scene_no_overlap.add_event(
            event_type="static",
            filepath=new_event_audio,
            alias="bad_event",
            emitter_kwargs=dict(keep_existing=True),
            event_kwargs=event_kwargs,
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
            emitter_kwargs=dict(keep_existing=True),
            event_kwargs=event_kwargs,
        )


@pytest.mark.parametrize(
    "new_event_audio,new_event_kws",
    [
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(),
        ),  # no custom event_start/duration, should be set automatically
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(event_start=15, duration=20),
        ),
        (utils_tests.SOUNDEVENT_DIR / "music/000010.mp3", dict(duration=5.0)),
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=10.0, duration=5.0, event_start=5.0),
        ),  # no overlap
    ],
)
def test_add_acceptable_event(
    new_event_audio, new_event_kws, oyens_scene_no_overlap: Scene
):
    """
    Test adding an acceptable event to a scene that already has events; should not be rejected
    """
    # Add the dummy event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="static",
        filepath=utils_tests.SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=dict(scene_start=5.0, duration=5.0),
    )
    # Add the tester event in: should not raise any errors
    oyens_scene_no_overlap.add_event(
        event_type="static",
        filepath=new_event_audio,
        alias="good_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=new_event_kws,
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
        (
            [utils_tests.SOUNDEVENT_DIR / "music"],
            False,
        ),  # this folder has some audio files inside it, so it's all good
        (None, ValueError),  # as if we've not provided `fp_path` to `Scene.__init__`
        (
            [utils.get_project_root() / "tests"],
            FileNotFoundError,
        ),  # no audio files inside this folder!
    ],
)
def test_get_random_foreground_audio(
    fg_path: str, raises, oyens_scene_no_overlap: Scene
):
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


def test_get_funcs(oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.add_event(event_type="static")
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
    events = oyens_scene_no_overlap.get_events()
    assert isinstance(events, list)
    assert isinstance(events[0], Event)

    # Trying to get event that doesn't exist will raise an error
    with pytest.raises(KeyError):
        oyens_scene_no_overlap.get_event("not_existing")


def test_clear_funcs(oyens_scene_no_overlap: Scene):
    # Add an event
    oyens_scene_no_overlap.add_event(event_type="static", alias="remover")
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

    oyens_scene_no_overlap.clear_emitters()
    oyens_scene_no_overlap.clear_microphones()
    assert len(oyens_scene_no_overlap.state.microphones) == 0
    assert len(oyens_scene_no_overlap.state.emitters) == 0

    oyens_scene_no_overlap.add_event(event_type="static", alias="remover")
    oyens_scene_no_overlap.clear_emitter(alias="remover")
    oyens_scene_no_overlap.add_event(event_type="static", alias="remover")
    oyens_scene_no_overlap.clear_emitters()
    assert len(oyens_scene_no_overlap.state.emitters) == 0

    oyens_scene_no_overlap.add_microphone(alias="remover")
    oyens_scene_no_overlap.clear_microphone(alias="remover")
    assert len(oyens_scene_no_overlap.state.microphones) == 0


@pytest.mark.parametrize("n_events", [1, 2, 3])
def test_generate(n_events: int, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_events()
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(
            event_type="static", emitter_kwargs=dict(keep_existing=True)
        )

    oyens_scene_no_overlap.generate("tmp.wav", "tmp.json")

    for fout in ["tmp.wav", "tmp.json"]:
        assert os.path.isfile(fout)
        os.remove(fout)


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


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "audiblelight_version": "0.1.0",
            "rlr_audio_propagation_version": "0.0.1",
            "creation_time": "2025-08-11_13:07:21",
            "duration": 50.0,
            "ref_db": -50,
            "max_overlap": 1,
            "fg_path": str(utils_tests.SOUNDEVENT_DIR),
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
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 5.0,
                    "scene_end": 10.0,
                    "event_start": 5.0,
                    "event_end": 10.0,
                    "duration": 5.0,
                    "snr": 5.185405340406351,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [-0.5, -0.5, 0.5],
                    "end_coordinates": [-0.5, -0.5, 0.5],
                    "emitters": [
                        {
                            "alias": "test_event",
                            "coordinates_absolute": [-0.5, -0.5, 0.5],
                        }
                    ],
                }
            },
            "state": {
                "emitters": {
                    "test_event": [
                        {
                            "alias": "test_event",
                            "coordinates_absolute": [-0.5, -0.5, 0.5],
                        }
                    ]
                },
                "microphones": {
                    "mic000": {
                        "name": "ambeovr",
                        "micarray_type": "AmbeoVR",
                        "is_spherical": True,
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
        },
        {
            "audiblelight_version": "0.1.0",
            "rlr_audio_propagation_version": "0.0.1",
            "creation_time": "2025-08-11_13:16:59",
            "duration": 30.0,
            "ref_db": -50,
            "max_overlap": 3,
            "fg_path": str(utils_tests.SOUNDEVENT_DIR),
            "ambience": {
                "ambience000": {
                    "alias": "ambience000",
                    "beta": 0,
                    "filepath": None,
                    "channels": 4,
                    "sample_rate": 44100,
                    "duration": 30.0,
                    "ref_db": -50,
                    "noise_kwargs": {},
                },
                "ambience001": {
                    "alias": "ambience001",
                    "beta": None,
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "waterTap/95709.wav"),
                    "channels": 4,
                    "sample_rate": 44100,
                    "duration": 30.0,
                    "ref_db": -50,
                    "noise_kwargs": {},
                },
            },
            "events": {
                "event000": {
                    "alias": "event000",
                    "filename": "001666.mp3",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "music/001666.mp3"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 7.981770941756986,
                    "scene_end": 37.95834690547581,
                    "event_start": 0.0,
                    "event_end": 29.976575963718822,
                    "duration": 29.976575963718822,
                    "snr": 25.74183954703098,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        -2.0909982476529274,
                        -1.4205935238035057,
                        1.3449372761851106,
                    ],
                    "end_coordinates": [
                        -2.0909982476529274,
                        -1.4205935238035057,
                        1.3449372761851106,
                    ],
                    "emitters": [
                        {
                            "alias": "event000",
                            "coordinates_absolute": [
                                -2.0909982476529274,
                                -1.4205935238035057,
                                1.3449372761851106,
                            ],
                        }
                    ],
                },
                "event001": {
                    "alias": "event001",
                    "filename": "236657.wav",
                    "filepath": str(
                        utils_tests.SOUNDEVENT_DIR / "femaleSpeech/236657.wav"
                    ),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 3.5278965484509284,
                    "scene_end": 3.9466947343919716,
                    "event_start": 0.0,
                    "event_end": 0.41879818594104307,
                    "duration": 0.41879818594104307,
                    "snr": 27.147279622962323,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        5.007669276685398,
                        -8.361759027155335,
                        1.4455039532646694,
                    ],
                    "end_coordinates": [
                        5.007669276685398,
                        -8.361759027155335,
                        1.4455039532646694,
                    ],
                    "emitters": [
                        {
                            "alias": "event001",
                            "coordinates_absolute": [
                                5.007669276685398,
                                -8.361759027155335,
                                1.4455039532646694,
                            ],
                        }
                    ],
                },
                "event002": {
                    "alias": "event002",
                    "filename": "007527.mp3",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "music/007527.mp3"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 12.850780138007437,
                    "scene_end": 42.82735610172626,
                    "event_start": 0.0,
                    "event_end": 29.976575963718822,
                    "duration": 29.976575963718822,
                    "snr": 13.710602163187696,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        4.640033609240403,
                        -8.023217530591952,
                        1.8839046478175203,
                    ],
                    "end_coordinates": [
                        4.640033609240403,
                        -8.023217530591952,
                        1.8839046478175203,
                    ],
                    "emitters": [
                        {
                            "alias": "event002",
                            "coordinates_absolute": [
                                4.640033609240403,
                                -8.023217530591952,
                                1.8839046478175203,
                            ],
                        }
                    ],
                },
                "event003": {
                    "alias": "event003",
                    "filename": "240693.wav",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "waterTap/240693.wav"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 0.8018063625684461,
                    "scene_end": 6.792214525833753,
                    "event_start": 0.0,
                    "event_end": 5.990408163265307,
                    "duration": 5.990408163265307,
                    "snr": 17.327370082020515,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        0.9494821839439869,
                        -5.062019458493836,
                        1.8257695048924627,
                    ],
                    "end_coordinates": [
                        0.9494821839439869,
                        -5.062019458493836,
                        1.8257695048924627,
                    ],
                    "emitters": [
                        {
                            "alias": "event003",
                            "coordinates_absolute": [
                                0.9494821839439869,
                                -5.062019458493836,
                                1.8257695048924627,
                            ],
                        }
                    ],
                },
                "event004": {
                    "alias": "event004",
                    "filename": "240693.wav",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "waterTap/240693.wav"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": True,
                    "scene_start": 28.510842827640893,
                    "scene_end": 34.5012509909062,
                    "event_start": 0.0,
                    "event_end": 5.990408163265307,
                    "duration": 5.990408163265307,
                    "snr": 25.298841093411685,
                    "sample_rate": 44100.0,
                    "spatial_resolution": 3.2045814626315168,
                    "spatial_velocity": 1.4147560194275908,
                    "start_coordinates": [
                        2.1684091658854454,
                        -4.0836262748673375,
                        1.063761155368002,
                    ],
                    "end_coordinates": [
                        5.487205489589439,
                        -1.5558905466926776,
                        1.2825129818748777,
                    ],
                    "emitters": [
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.1684091658854454,
                                -4.0836262748673375,
                                1.063761155368002,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.3430826566067084,
                                -3.9505875523318292,
                                1.0752744093946796,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.517756147327971,
                                -3.817548829796321,
                                1.0867876634213574,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.6924296380492336,
                                -3.684510107260812,
                                1.098300917448035,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.8671031287704967,
                                -3.551471384725304,
                                1.1098141714747127,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.0417766194917597,
                                -3.4184326621897956,
                                1.1213274255013903,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.2164501102130223,
                                -3.285393939654287,
                                1.1328406795280679,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.391123600934285,
                                -3.1523552171187785,
                                1.1443539335547457,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.565797091655548,
                                -3.0193164945832702,
                                1.1558671875814233,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.740470582376811,
                                -2.886277772047762,
                                1.167380441608101,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.9151440730980736,
                                -2.7532390495122536,
                                1.1788936956347786,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.089817563819336,
                                -2.620200326976745,
                                1.1904069496614564,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.264491054540599,
                                -2.4871616044412366,
                                1.201920203688134,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.439164545261862,
                                -2.3541228819057283,
                                1.2134334577148116,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.613838035983125,
                                -2.2210841593702195,
                                1.2249467117414894,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.788511526704387,
                                -2.088045436834711,
                                1.236459965768167,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.96318501742565,
                                -1.955006714299203,
                                1.2479732197948448,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                5.1378585081469135,
                                -1.8219679917636946,
                                1.2594864738215223,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                5.312531998868176,
                                -1.6889292692281863,
                                1.2709997278482001,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                5.487205489589439,
                                -1.5558905466926776,
                                1.2825129818748777,
                            ],
                        },
                    ],
                },
            },
            "state": {
                "emitters": {
                    "event000": [
                        {
                            "alias": "event000",
                            "coordinates_absolute": [
                                -2.0909982476529274,
                                -1.4205935238035057,
                                1.3449372761851106,
                            ],
                        }
                    ],
                    "event001": [
                        {
                            "alias": "event001",
                            "coordinates_absolute": [
                                5.007669276685398,
                                -8.361759027155335,
                                1.4455039532646694,
                            ],
                        }
                    ],
                    "event002": [
                        {
                            "alias": "event002",
                            "coordinates_absolute": [
                                4.640033609240403,
                                -8.023217530591952,
                                1.8839046478175203,
                            ],
                        }
                    ],
                    "event003": [
                        {
                            "alias": "event003",
                            "coordinates_absolute": [
                                0.9494821839439869,
                                -5.062019458493836,
                                1.8257695048924627,
                            ],
                        }
                    ],
                    "event004": [
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.1684091658854454,
                                -4.0836262748673375,
                                1.063761155368002,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.3430826566067084,
                                -3.9505875523318292,
                                1.0752744093946796,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.517756147327971,
                                -3.817548829796321,
                                1.0867876634213574,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.6924296380492336,
                                -3.684510107260812,
                                1.098300917448035,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                2.8671031287704967,
                                -3.551471384725304,
                                1.1098141714747127,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.0417766194917597,
                                -3.4184326621897956,
                                1.1213274255013903,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.2164501102130223,
                                -3.285393939654287,
                                1.1328406795280679,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.391123600934285,
                                -3.1523552171187785,
                                1.1443539335547457,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.565797091655548,
                                -3.0193164945832702,
                                1.1558671875814233,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.740470582376811,
                                -2.886277772047762,
                                1.167380441608101,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                3.9151440730980736,
                                -2.7532390495122536,
                                1.1788936956347786,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.089817563819336,
                                -2.620200326976745,
                                1.1904069496614564,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.264491054540599,
                                -2.4871616044412366,
                                1.201920203688134,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.439164545261862,
                                -2.3541228819057283,
                                1.2134334577148116,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.613838035983125,
                                -2.2210841593702195,
                                1.2249467117414894,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.788511526704387,
                                -2.088045436834711,
                                1.236459965768167,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                4.96318501742565,
                                -1.955006714299203,
                                1.2479732197948448,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                5.1378585081469135,
                                -1.8219679917636946,
                                1.2594864738215223,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                5.312531998868176,
                                -1.6889292692281863,
                                1.2709997278482001,
                            ],
                        },
                        {
                            "alias": "event004",
                            "coordinates_absolute": [
                                5.487205489589439,
                                -1.5558905466926776,
                                1.2825129818748777,
                            ],
                        },
                    ],
                },
                "microphones": {
                    "tetra": {
                        "name": "ambeovr",
                        "micarray_type": "AmbeoVR",
                        "is_spherical": True,
                        "n_capsules": 4,
                        "capsule_names": ["FLU", "FRD", "BLD", "BRU"],
                        "coordinates_absolute": [
                            [3.704459358637966, -6.904224745071837, 1.3144022071631645],
                            [3.704459358637966, -6.915809304378629, 1.3029306784361434],
                            [3.692874799331175, -6.904224745071837, 1.3029306784361434],
                            [3.692874799331175, -6.915809304378629, 1.3144022071631645],
                        ],
                        "coordinates_center": [
                            3.6986670789845704,
                            -6.910017024725233,
                            1.308666442799654,
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
            "spatial_format": "A",
        },
    ],
)
def test_scene_from_dict(input_dict: dict):
    ev = Scene.from_dict(input_dict)
    assert isinstance(ev, Scene)
    assert len(ev.events) == len(input_dict["events"])
    assert (
        sum(len(em) for em in ev.state.emitters.values())
        == sum(len(em) for em in input_dict["state"]["emitters"].values())
        == ev.state.ctx.get_source_count()
    )
    assert len(ev.ambience.keys()) == len(input_dict["ambience"])


@pytest.mark.parametrize(
    "filepath, emitter_kws, event_kws",
    [
        # Test 1: explicitly define a filepath, emitter keywords, and event keywords (overrides)
        (
            utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            dict(
                position=np.array([-0.5, -0.5, 0.5]),
                polar=False,
                ensure_direct_path=False,
            ),
            dict(duration=5, event_start=5, scene_start=5),
        ),
    ],
)
def test_magic_methods(filepath, emitter_kws, event_kws, oyens_scene_no_overlap):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="static",
        filepath=filepath,
        alias="test_event",
        emitter_kwargs=emitter_kws,
        event_kwargs=event_kws,
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
