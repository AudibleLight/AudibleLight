#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py"""

import os
from pathlib import Path

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.augmentation import LowpassFilter, Phaser, TimeShift
from audiblelight.core import Scene
from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import Emitter
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
            augmentations=TimeShift(stretch_factor=0.5),
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
            shape="circular",
            filepath=utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            augmentations=TimeShift,
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
        # Test with a polar starting position
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
    ],
)
def test_add_moving_event(kwargs, oyens_scene_no_overlap: Scene):
    # Clear out the randomly-added mic and add one in a specific position (for reproducibility)
    #  This is necessary because in the final test, we assume a polar position WRT the mic
    #  But if the mic is placed randomly, this polar position can sometimes be invalid
    is_polar = kwargs.get("polar", False)
    if is_polar:
        oyens_scene_no_overlap.clear_microphones()
        oyens_scene_no_overlap.add_microphone(alias="mic000", position=[2.5, -1.0, 1.0])

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
def test_generate(n_events: int, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_events()
    for n_event in range(n_events):
        # Use a short duration here so we don't run into issues with placing events
        oyens_scene_no_overlap.add_event(event_type="static", duration=1.0)

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
                    "emitters": [
                        [1.8156068957785347, -1.863507837016133, 1.8473540916136413]
                    ],
                    "emitters_relative": {
                        "mic000": [
                            [203.9109387558252, -5.976352087676762, 3.3744825372046803]
                        ]
                    },
                }
            },
            "state": {
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
        }
    ],
)
def test_scene_from_dict(input_dict: dict):
    ev = Scene.from_dict(input_dict)
    assert isinstance(ev, Scene)
    assert len(ev.events) == len(input_dict["events"])
    assert (
        ev.state.num_emitters
        == sum(len(em) for em in input_dict["state"]["emitters"].values())
        == ev.state.ctx.get_source_count()
    )
    assert len(ev.ambience.keys()) == len(input_dict["ambience"])


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
        ([Phaser, LowpassFilter, TimeShift], 2),
        ([Phaser], 1),
        # Coerced down to 2
        ([Phaser, LowpassFilter], 5000),
    ],
)
@pytest.mark.parametrize("event_type", ["static", "moving"])
def test_add_events_with_random_augmentations(aug_list, n_augs, event_type):
    """
    Add events with N random augmentations, drawn from our list
    """
    sc = Scene(
        duration=50,
        mesh_path=utils_tests.OYENS_PATH,
        event_augmentations=aug_list,
        fg_path=utils_tests.SOUNDEVENT_DIR,
        max_overlap=1,
    )
    ev = sc.add_event(augmentations=n_augs, event_type=event_type)
    augs = ev.get_augmentations()

    # All augs should be sampled from our list
    for aug in augs:
        assert type(aug) in aug_list

    # Augmentations should always be unique when random sampling
    assert len(augs) == len(list(set(augs)))

    # Should have expected number of augs
    assert len(augs) == min(len(sc.event_augmentations), n_augs)
