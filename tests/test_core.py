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
                        [-0.5, -0.5, 0.5],
                    ],
                }
            },
            "state": {
                "emitters": {
                    "test_event": [
                        [-0.5, -0.5, 0.5],
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
            "creation_time": "2025-08-11_14:21:13",
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
                    "filename": "93856.wav",
                    "filepath": str(
                        utils_tests.SOUNDEVENT_DIR / "maleSpeech/93856.wav"
                    ),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 8.812249505679185,
                    "scene_end": 9.265038621325443,
                    "event_start": 0.0,
                    "event_end": 0.4527891156462585,
                    "duration": 0.4527891156462585,
                    "snr": 29.029578165525322,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        2.75861354993879,
                        -1.6199735396985329,
                        0.4482871425244255,
                    ],
                    "end_coordinates": [
                        2.75861354993879,
                        -1.6199735396985329,
                        0.4482871425244255,
                    ],
                    "num_emitters": 1,
                    "emitters": [
                        [2.75861354993879, -1.6199735396985329, 0.4482871425244255]
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
                    "scene_start": 9.33668388157278,
                    "scene_end": 9.755482067513823,
                    "event_start": 0.0,
                    "event_end": 0.41879818594104307,
                    "duration": 0.41879818594104307,
                    "snr": 18.337663556181106,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        1.451795233185468,
                        -5.204843321294307,
                        1.1484658843042004,
                    ],
                    "end_coordinates": [
                        1.451795233185468,
                        -5.204843321294307,
                        1.1484658843042004,
                    ],
                    "num_emitters": 1,
                    "emitters": [
                        [1.451795233185468, -5.204843321294307, 1.1484658843042004]
                    ],
                },
                "event002": {
                    "alias": "event002",
                    "filename": "007527.mp3",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "music/007527.mp3"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 28.98229911181565,
                    "scene_end": 58.95887507553447,
                    "event_start": 0.0,
                    "event_end": 29.976575963718822,
                    "duration": 29.976575963718822,
                    "snr": 17.66172309231914,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        3.7954463446663036,
                        -5.454712940913788,
                        1.6274903797382563,
                    ],
                    "end_coordinates": [
                        3.7954463446663036,
                        -5.454712940913788,
                        1.6274903797382563,
                    ],
                    "num_emitters": 1,
                    "emitters": [
                        [3.7954463446663036, -5.454712940913788, 1.6274903797382563]
                    ],
                },
                "event003": {
                    "alias": "event003",
                    "filename": "30085.wav",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "telephone/30085.wav"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": False,
                    "scene_start": 6.472908460600676,
                    "scene_end": 10.362250864228795,
                    "event_start": 0.0,
                    "event_end": 3.889342403628118,
                    "duration": 3.889342403628118,
                    "snr": 25.55806050533056,
                    "sample_rate": 44100.0,
                    "spatial_resolution": None,
                    "spatial_velocity": None,
                    "start_coordinates": [
                        4.37756516034859,
                        -7.559676309493261,
                        0.8787509777746272,
                    ],
                    "end_coordinates": [
                        4.37756516034859,
                        -7.559676309493261,
                        0.8787509777746272,
                    ],
                    "num_emitters": 1,
                    "emitters": [
                        [4.37756516034859, -7.559676309493261, 0.8787509777746272]
                    ],
                },
                "event004": {
                    "alias": "event004",
                    "filename": "240693.wav",
                    "filepath": str(utils_tests.SOUNDEVENT_DIR / "waterTap/240693.wav"),
                    "class_id": None,
                    "class_label": None,
                    "is_moving": True,
                    "scene_start": 21.304835303603674,
                    "scene_end": 27.295243466868982,
                    "event_start": 0.0,
                    "event_end": 5.990408163265307,
                    "duration": 5.990408163265307,
                    "snr": 26.31666181644687,
                    "sample_rate": 44100.0,
                    "spatial_resolution": 2.350573517115879,
                    "spatial_velocity": 0.5339076324864093,
                    "start_coordinates": [
                        2.261394920448118,
                        -0.21339953468371853,
                        1.0865180964455003,
                    ],
                    "end_coordinates": [
                        2.8767877662288184,
                        -0.3463581007031614,
                        1.838455595680327,
                    ],
                    "num_emitters": 5,
                    "emitters": [
                        [2.261394920448118, -0.21339953468371853, 1.0865180964455003],
                        [2.415243131893293, -0.24663917618857925, 1.274502471254207],
                        [2.569091343338468, -0.27987881769344, 1.4624868460629137],
                        [2.7229395547836432, -0.3131184591983007, 1.6504712208716206],
                        [2.8767877662288184, -0.3463581007031614, 1.838455595680327],
                    ],
                },
            },
            "state": {
                "emitters": {
                    "event000": [
                        [2.75861354993879, -1.6199735396985329, 0.4482871425244255]
                    ],
                    "event001": [
                        [1.451795233185468, -5.204843321294307, 1.1484658843042004]
                    ],
                    "event002": [
                        [3.7954463446663036, -5.454712940913788, 1.6274903797382563]
                    ],
                    "event003": [
                        [4.37756516034859, -7.559676309493261, 0.8787509777746272]
                    ],
                    "event004": [
                        [2.261394920448118, -0.21339953468371853, 1.0865180964455003],
                        [2.415243131893293, -0.24663917618857925, 1.274502471254207],
                        [2.569091343338468, -0.27987881769344, 1.4624868460629137],
                        [2.7229395547836432, -0.3131184591983007, 1.6504712208716206],
                        [2.8767877662288184, -0.3463581007031614, 1.838455595680327],
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
                            [
                                3.6105085415895757,
                                -8.384402443789977,
                                2.1526455364771224,
                            ],
                            [3.6105085415895757, -8.395987003096767, 2.141174007750102],
                            [3.5989239822827845, -8.384402443789977, 2.141174007750102],
                            [
                                3.5989239822827845,
                                -8.395987003096767,
                                2.1526455364771224,
                            ],
                        ],
                        "coordinates_center": [
                            3.60471626193618,
                            -8.390194723443372,
                            2.146909772113612,
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
