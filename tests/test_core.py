#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/core.py"""

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.stats as stats

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import Emitter

SOUNDEVENT_DIR = utils.get_project_root() / "tests/test_resources/soundevents"
MESH_DIR = utils.get_project_root() / "tests/test_resources/meshes"
OYENS_PATH = MESH_DIR / "Oyens.glb"


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
                ensure_direct_path=False,
            ),
            dict(duration=5, event_start=5, scene_start=5),
        ),
        # Test 2: explicit event keywords and filepath, but no emitter keywords
        (SOUNDEVENT_DIR / "music/001666.mp3", None, dict(snr=5, spatial_velocity=5)),
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
def test_add_event(
    filepath: str, emitter_kws, event_kws, oyens_scene_no_overlap: Scene
):
    # Add the event in
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
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
    "new_event_audio,event_kwargs,raises",
    [
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=6.0, duration=1.0),
            ValueError,
        ),
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=9.0, duration=10.0),
            ValueError,
        ),
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
            dict(scene_start=3.0, duration=10.0),
            ValueError,
        ),
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
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
        filepath=SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        event_kwargs=dict(scene_start=5.0, duration=5.0),
    )
    # Add the tester event in: should raise an error
    with pytest.raises(raises):
        oyens_scene_no_overlap.add_event(
            filepath=new_event_audio,
            alias="bad_event",
            emitter_kwargs=dict(keep_existing=True),
            event_kwargs=event_kwargs,
        )
    # Should not be added to the dictionary
    assert len(oyens_scene_no_overlap.events) == 1
    with pytest.raises(KeyError):
        _ = oyens_scene_no_overlap.get_event("bad_event")


@pytest.mark.parametrize(
    "new_event_audio,new_event_kws",
    [
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
            dict(),
        ),  # no custom event_start/duration, should be set automatically
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(event_start=15, duration=20)),
        (SOUNDEVENT_DIR / "music/000010.mp3", dict(duration=5.0)),
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
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
        filepath=SOUNDEVENT_DIR / "music/001666.mp3",
        alias="dummy_event",
        emitter_kwargs=dict(keep_existing=True),
        event_kwargs=dict(scene_start=5.0, duration=5.0),
    )
    # Add the tester event in: should not raise any errors
    oyens_scene_no_overlap.add_event(
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
            [SOUNDEVENT_DIR / "music"],
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
    events = oyens_scene_no_overlap.get_events()
    assert isinstance(events, list)
    assert isinstance(events[0], Event)

    # Trying to get event that doesn't exist will raise an error
    with pytest.raises(KeyError):
        oyens_scene_no_overlap.get_event("not_existing")


def test_clear_funcs(oyens_scene_no_overlap: Scene):
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

    oyens_scene_no_overlap.clear_emitters()
    oyens_scene_no_overlap.clear_microphones()
    assert len(oyens_scene_no_overlap.state.microphones) == 0
    assert len(oyens_scene_no_overlap.state.emitters) == 0

    oyens_scene_no_overlap.add_event(alias="remover")
    oyens_scene_no_overlap.clear_emitter(alias="remover")
    oyens_scene_no_overlap.add_event(alias="remover")
    oyens_scene_no_overlap.clear_emitters()
    assert len(oyens_scene_no_overlap.state.emitters) == 0

    oyens_scene_no_overlap.add_microphone(alias="remover")
    oyens_scene_no_overlap.clear_microphone(alias="remover")
    assert len(oyens_scene_no_overlap.state.microphones) == 0


@pytest.mark.parametrize("n_events", [1, 2, 3])
def test_generate(n_events: int, oyens_scene_no_overlap: Scene):
    oyens_scene_no_overlap.clear_events()
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(emitter_kwargs=dict(keep_existing=True))

    oyens_scene_no_overlap.generate("tmp.wav", "tmp.json")

    for fout in ["tmp.wav", "tmp.json"]:
        assert os.path.isfile(fout)
        os.remove(fout)


@pytest.mark.parametrize(
    "noise, filepath",
    [
        ("white", None),
        (2.0, None),
        (
            None,
            utils.get_project_root()
            / "tests/test_resources/soundevents/waterTap/95709.wav",
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
    "mesh_fpath",
    [
        MESH_DIR / mesh for mesh in os.listdir(MESH_DIR)
    ],  # run on every mesh we have in the test directory
)
@pytest.mark.parametrize("mic_type", ["ambeovr", "eigenmike32"])
@pytest.mark.parametrize("n_events, duration, max_overlap", [(1, 30, 3), (9, 50, 6)])
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_pipeline(mesh_fpath, n_events, duration, max_overlap, mic_type):
    """
    This function tests the whole pipeline, from generating a Scene with a given mesh, adding events, and creating audio
    """
    # Create the scene
    sc = Scene(
        duration=duration,
        mesh_path=mesh_fpath,
        # Pass some default distributions for everything
        event_start_dist=stats.uniform(0, 10),
        event_duration_dist=stats.uniform(0, 10),
        event_velocity_dist=stats.uniform(0, 10),
        event_resolution_dist=stats.uniform(0, 10),
        snr_dist=stats.norm(5, 1),
        fg_path=SOUNDEVENT_DIR,
        max_overlap=max_overlap,
    )
    # Add the desired microphone type and number of events
    sc.add_microphone(microphone_type=mic_type)
    for i in range(n_events):
        sc.add_event(emitter_kwargs=dict(keep_existing=True))
    # Generate everything and check the files exist
    sc.generate(audio_path="audio_out.wav", metadata_path="metadata_out.json")
    for path in ["audio_out.wav", "metadata_out.json"]:
        assert os.path.isfile(path)
        os.remove(path)


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "audiblelight_version": "0.1.0",
            "rlr_audio_propagation_version": "0.0.1",
            "creation_time": "2025-07-29_15:32:49",
            "duration": 30.0,
            "ref_db": -50,
            "max_overlap": 3.0,
            "fg_path": str(
                utils.get_project_root() / "tests/test_resources/soundevents"
            ),
            "ambience": {
                "tester": {
                    "alias": "tester",
                    "beta": 1,
                    "filepath": None,
                    "channels": 4,
                    "sample_rate": 44100.0,
                    "duration": 10.0,
                    "ref_db": -65,
                    "noise_kwargs": {},
                },
                "tester_audio": {
                    "alias": "tester_audio",
                    "beta": None,
                    "filepath": utils.get_project_root()
                    / "tests/test_resources/soundevents/waterTap/95709.wav",
                    "channels": 4,
                    "sample_rate": 44100.0,
                    "duration": 10.0,
                    "ref_db": -65,
                    "noise_kwargs": {},
                },
            },
            "events": {
                "event000": {
                    "alias": "event000",
                    "filename": "000010.mp3",
                    "filepath": str(
                        utils.get_project_root()
                        / "tests/test_resources/soundevents/music/000010.mp3"
                    ),
                    "class_id": None,
                    "class_label": None,
                    "scene_start": 3.3414165656885886,
                    "scene_end": 12.31872357200817,
                    "event_start": 9.80077281297744,
                    "event_end": 18.77807981929702,
                    "duration": 8.977307006319581,
                    "snr": 6.358210053654571,
                    "sample_rate": 44100.0,
                    "spatial_resolution": 9.608687144731517,
                    "spatial_velocity": 1.076633223976352,
                    "start_coordinates": {
                        "absolute": [
                            3.138291668058967,
                            0.038534063951257025,
                            2.0482037990906727,
                        ],
                        "relative_cartesian": {
                            "event000": [0.0, 0.0, 0.0],
                            "mic000": [
                                -0.8859426619553759,
                                2.9534221529206377,
                                0.36002469289039984,
                            ],
                            "event001": [
                                [
                                    -2.2333563194364796,
                                    -1.4665946631472622,
                                    0.3130778986929861,
                                ]
                            ],
                            "event002": [
                                [
                                    -1.7106695658740882,
                                    5.0752318593323515,
                                    -0.07563942967148884,
                                ]
                            ],
                        },
                        "relative_polar": {
                            "event000": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    106.69774947852416,
                                    83.3402562971731,
                                    3.104386347271515,
                                ]
                            ],
                            "event001": [
                                [
                                    213.29200206063442,
                                    83.31675945948695,
                                    2.6901297601024576,
                                ]
                            ],
                            "event002": [
                                [
                                    108.62702433583053,
                                    90.80913196273397,
                                    5.356313108184676,
                                ]
                            ],
                        },
                    },
                    "end_coordinates": {
                        "absolute": [
                            3.138291668058967,
                            0.038534063951257025,
                            2.0482037990906727,
                        ],
                        "relative_cartesian": {
                            "event000": [0.0, 0.0, 0.0],
                            "mic000": [
                                -0.8859426619553759,
                                2.9534221529206377,
                                0.36002469289039984,
                            ],
                            "event001": [
                                [
                                    -2.2333563194364796,
                                    -1.4665946631472622,
                                    0.3130778986929861,
                                ]
                            ],
                            "event002": [
                                [
                                    -1.7106695658740882,
                                    5.0752318593323515,
                                    -0.07563942967148884,
                                ]
                            ],
                        },
                        "relative_polar": {
                            "event000": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    106.69774947852416,
                                    83.3402562971731,
                                    3.104386347271515,
                                ]
                            ],
                            "event001": [
                                [
                                    213.29200206063442,
                                    83.31675945948695,
                                    2.6901297601024576,
                                ]
                            ],
                            "event002": [
                                [
                                    108.62702433583053,
                                    90.80913196273397,
                                    5.356313108184676,
                                ]
                            ],
                        },
                    },
                    "emitters": [
                        {
                            "alias": "event000",
                            "coordinates_absolute": [
                                3.138291668058967,
                                0.038534063951257025,
                                2.0482037990906727,
                            ],
                            "coordinates_relative_cartesian": {
                                "event000": [0.0, 0.0, 0.0],
                                "mic000": [
                                    -0.8859426619553759,
                                    2.9534221529206377,
                                    0.36002469289039984,
                                ],
                                "event001": [
                                    [
                                        -2.2333563194364796,
                                        -1.4665946631472622,
                                        0.3130778986929861,
                                    ]
                                ],
                                "event002": [
                                    [
                                        -1.7106695658740882,
                                        5.0752318593323515,
                                        -0.07563942967148884,
                                    ]
                                ],
                            },
                            "coordinates_relative_polar": {
                                "event000": [0.0, 0.0, 0.0],
                                "mic000": [
                                    [
                                        106.69774947852416,
                                        83.3402562971731,
                                        3.104386347271515,
                                    ]
                                ],
                                "event001": [
                                    [
                                        213.29200206063442,
                                        83.31675945948695,
                                        2.6901297601024576,
                                    ]
                                ],
                                "event002": [
                                    [
                                        108.62702433583053,
                                        90.80913196273397,
                                        5.356313108184676,
                                    ]
                                ],
                            },
                        }
                    ],
                },
                "event001": {
                    "alias": "event001",
                    "filename": "431669.wav",
                    "filepath": str(
                        utils.get_project_root()
                        / "tests/test_resources/soundevents/telephone/431669.wav"
                    ),
                    "class_id": None,
                    "class_label": None,
                    "scene_start": 17.921930748109894,
                    "scene_end": 20.96372213132985,
                    "event_start": 0.0,
                    "event_end": 3.041791383219955,
                    "duration": 3.041791383219955,
                    "snr": 3.3007158402687256,
                    "sample_rate": 44100.0,
                    "spatial_resolution": 1.6614832228881637,
                    "spatial_velocity": 7.397699882455427,
                    "start_coordinates": {
                        "absolute": [
                            5.371647987495447,
                            1.5051287270985192,
                            1.7351259003976867,
                        ],
                        "relative_cartesian": {
                            "event000": [
                                [
                                    2.2333563194364796,
                                    1.4665946631472622,
                                    -0.3130778986929861,
                                ]
                            ],
                            "event001": [0.0, 0.0, 0.0],
                            "mic000": [
                                1.3474136574811038,
                                4.4200168160679,
                                0.046946794197413766,
                            ],
                            "event002": [
                                [
                                    0.5226867535623914,
                                    6.541826522479614,
                                    -0.3887173283644749,
                                ]
                            ],
                        },
                        "relative_polar": {
                            "event000": [
                                [
                                    33.29200206063442,
                                    96.68324054051307,
                                    2.6901297601024576,
                                ]
                            ],
                            "event001": [0.0, 0.0, 0.0],
                            "mic000": [
                                [73.04649501893131, 89.41790533787594, 4.62106873138401]
                            ],
                            "event002": [
                                [
                                    85.43181705637195,
                                    93.38975691121261,
                                    6.574176515270801,
                                ]
                            ],
                        },
                    },
                    "end_coordinates": {
                        "absolute": [
                            5.371647987495447,
                            1.5051287270985192,
                            1.7351259003976867,
                        ],
                        "relative_cartesian": {
                            "event000": [
                                [
                                    2.2333563194364796,
                                    1.4665946631472622,
                                    -0.3130778986929861,
                                ]
                            ],
                            "event001": [0.0, 0.0, 0.0],
                            "mic000": [
                                1.3474136574811038,
                                4.4200168160679,
                                0.046946794197413766,
                            ],
                            "event002": [
                                [
                                    0.5226867535623914,
                                    6.541826522479614,
                                    -0.3887173283644749,
                                ]
                            ],
                        },
                        "relative_polar": {
                            "event000": [
                                [
                                    33.29200206063442,
                                    96.68324054051307,
                                    2.6901297601024576,
                                ]
                            ],
                            "event001": [0.0, 0.0, 0.0],
                            "mic000": [
                                [73.04649501893131, 89.41790533787594, 4.62106873138401]
                            ],
                            "event002": [
                                [
                                    85.43181705637195,
                                    93.38975691121261,
                                    6.574176515270801,
                                ]
                            ],
                        },
                    },
                    "emitters": [
                        {
                            "alias": "event001",
                            "coordinates_absolute": [
                                5.371647987495447,
                                1.5051287270985192,
                                1.7351259003976867,
                            ],
                            "coordinates_relative_cartesian": {
                                "event000": [
                                    [
                                        2.2333563194364796,
                                        1.4665946631472622,
                                        -0.3130778986929861,
                                    ]
                                ],
                                "event001": [0.0, 0.0, 0.0],
                                "mic000": [
                                    1.3474136574811038,
                                    4.4200168160679,
                                    0.046946794197413766,
                                ],
                                "event002": [
                                    [
                                        0.5226867535623914,
                                        6.541826522479614,
                                        -0.3887173283644749,
                                    ]
                                ],
                            },
                            "coordinates_relative_polar": {
                                "event000": [
                                    [
                                        33.29200206063442,
                                        96.68324054051307,
                                        2.6901297601024576,
                                    ]
                                ],
                                "event001": [0.0, 0.0, 0.0],
                                "mic000": [
                                    [
                                        73.04649501893131,
                                        89.41790533787594,
                                        4.62106873138401,
                                    ]
                                ],
                                "event002": [
                                    [
                                        85.43181705637195,
                                        93.38975691121261,
                                        6.574176515270801,
                                    ]
                                ],
                            },
                        }
                    ],
                },
                "event002": {
                    "alias": "event002",
                    "filename": "93899.wav",
                    "filepath": str(
                        utils.get_project_root()
                        / "tests/test_resources/soundevents/maleSpeech/93899.wav"
                    ),
                    "class_id": None,
                    "class_label": None,
                    "scene_start": 15.273160864969604,
                    "scene_end": 15.760779912588651,
                    "event_start": 0.0,
                    "event_end": 0.4876190476190476,
                    "duration": 0.4876190476190476,
                    "snr": 5.284228284497229,
                    "sample_rate": 44100.0,
                    "spatial_resolution": 5.414582517062746,
                    "spatial_velocity": 8.762996314561452,
                    "start_coordinates": {
                        "absolute": [
                            4.848961233933055,
                            -5.0366977953810945,
                            2.1238432287621616,
                        ],
                        "relative_cartesian": {
                            "event000": [
                                [
                                    1.7106695658740882,
                                    -5.0752318593323515,
                                    0.07563942967148884,
                                ]
                            ],
                            "event001": [
                                [
                                    -0.5226867535623914,
                                    -6.541826522479614,
                                    0.3887173283644749,
                                ]
                            ],
                            "event002": [0.0, 0.0, 0.0],
                            "mic000": [
                                0.8247269039187124,
                                -2.121809706411714,
                                0.4356641225618887,
                            ],
                        },
                        "relative_polar": {
                            "event000": [
                                [
                                    288.6270243358305,
                                    89.19086803726604,
                                    5.356313108184676,
                                ]
                            ],
                            "event001": [
                                [265.431817056372, 86.6102430887874, 6.574176515270801]
                            ],
                            "event002": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    291.24062250983945,
                                    79.16583571951304,
                                    2.317769212833307,
                                ]
                            ],
                        },
                    },
                    "end_coordinates": {
                        "absolute": [
                            4.848961233933055,
                            -5.0366977953810945,
                            2.1238432287621616,
                        ],
                        "relative_cartesian": {
                            "event000": [
                                [
                                    1.7106695658740882,
                                    -5.0752318593323515,
                                    0.07563942967148884,
                                ]
                            ],
                            "event001": [
                                [
                                    -0.5226867535623914,
                                    -6.541826522479614,
                                    0.3887173283644749,
                                ]
                            ],
                            "event002": [0.0, 0.0, 0.0],
                            "mic000": [
                                0.8247269039187124,
                                -2.121809706411714,
                                0.4356641225618887,
                            ],
                        },
                        "relative_polar": {
                            "event000": [
                                [
                                    288.6270243358305,
                                    89.19086803726604,
                                    5.356313108184676,
                                ]
                            ],
                            "event001": [
                                [265.431817056372, 86.6102430887874, 6.574176515270801]
                            ],
                            "event002": [0.0, 0.0, 0.0],
                            "mic000": [
                                [
                                    291.24062250983945,
                                    79.16583571951304,
                                    2.317769212833307,
                                ]
                            ],
                        },
                    },
                    "emitters": [
                        {
                            "alias": "event002",
                            "coordinates_absolute": [
                                4.848961233933055,
                                -5.0366977953810945,
                                2.1238432287621616,
                            ],
                            "coordinates_relative_cartesian": {
                                "event000": [
                                    [
                                        1.7106695658740882,
                                        -5.0752318593323515,
                                        0.07563942967148884,
                                    ]
                                ],
                                "event001": [
                                    [
                                        -0.5226867535623914,
                                        -6.541826522479614,
                                        0.3887173283644749,
                                    ]
                                ],
                                "event002": [0.0, 0.0, 0.0],
                                "mic000": [
                                    0.8247269039187124,
                                    -2.121809706411714,
                                    0.4356641225618887,
                                ],
                            },
                            "coordinates_relative_polar": {
                                "event000": [
                                    [
                                        288.6270243358305,
                                        89.19086803726604,
                                        5.356313108184676,
                                    ]
                                ],
                                "event001": [
                                    [
                                        265.431817056372,
                                        86.6102430887874,
                                        6.574176515270801,
                                    ]
                                ],
                                "event002": [0.0, 0.0, 0.0],
                                "mic000": [
                                    [
                                        291.24062250983945,
                                        79.16583571951304,
                                        2.317769212833307,
                                    ]
                                ],
                            },
                        }
                    ],
                },
            },
            "state": {
                "emitters": {
                    "event000": [
                        {
                            "alias": "event000",
                            "coordinates_absolute": [
                                3.138291668058967,
                                0.038534063951257025,
                                2.0482037990906727,
                            ],
                            "coordinates_relative_cartesian": {
                                "event000": [0.0, 0.0, 0.0],
                                "mic000": [
                                    -0.8859426619553759,
                                    2.9534221529206377,
                                    0.36002469289039984,
                                ],
                                "event001": [
                                    [
                                        -2.2333563194364796,
                                        -1.4665946631472622,
                                        0.3130778986929861,
                                    ]
                                ],
                                "event002": [
                                    [
                                        -1.7106695658740882,
                                        5.0752318593323515,
                                        -0.07563942967148884,
                                    ]
                                ],
                            },
                            "coordinates_relative_polar": {
                                "event000": [0.0, 0.0, 0.0],
                                "mic000": [
                                    [
                                        106.69774947852416,
                                        83.3402562971731,
                                        3.104386347271515,
                                    ]
                                ],
                                "event001": [
                                    [
                                        213.29200206063442,
                                        83.31675945948695,
                                        2.6901297601024576,
                                    ]
                                ],
                                "event002": [
                                    [
                                        108.62702433583053,
                                        90.80913196273397,
                                        5.356313108184676,
                                    ]
                                ],
                            },
                        }
                    ],
                    "event001": [
                        {
                            "alias": "event001",
                            "coordinates_absolute": [
                                5.371647987495447,
                                1.5051287270985192,
                                1.7351259003976867,
                            ],
                            "coordinates_relative_cartesian": {
                                "event000": [
                                    [
                                        2.2333563194364796,
                                        1.4665946631472622,
                                        -0.3130778986929861,
                                    ]
                                ],
                                "event001": [0.0, 0.0, 0.0],
                                "mic000": [
                                    1.3474136574811038,
                                    4.4200168160679,
                                    0.046946794197413766,
                                ],
                                "event002": [
                                    [
                                        0.5226867535623914,
                                        6.541826522479614,
                                        -0.3887173283644749,
                                    ]
                                ],
                            },
                            "coordinates_relative_polar": {
                                "event000": [
                                    [
                                        33.29200206063442,
                                        96.68324054051307,
                                        2.6901297601024576,
                                    ]
                                ],
                                "event001": [0.0, 0.0, 0.0],
                                "mic000": [
                                    [
                                        73.04649501893131,
                                        89.41790533787594,
                                        4.62106873138401,
                                    ]
                                ],
                                "event002": [
                                    [
                                        85.43181705637195,
                                        93.38975691121261,
                                        6.574176515270801,
                                    ]
                                ],
                            },
                        }
                    ],
                    "event002": [
                        {
                            "alias": "event002",
                            "coordinates_absolute": [
                                4.848961233933055,
                                -5.0366977953810945,
                                2.1238432287621616,
                            ],
                            "coordinates_relative_cartesian": {
                                "event000": [
                                    [
                                        1.7106695658740882,
                                        -5.0752318593323515,
                                        0.07563942967148884,
                                    ]
                                ],
                                "event001": [
                                    [
                                        -0.5226867535623914,
                                        -6.541826522479614,
                                        0.3887173283644749,
                                    ]
                                ],
                                "event002": [0.0, 0.0, 0.0],
                                "mic000": [
                                    0.8247269039187124,
                                    -2.121809706411714,
                                    0.4356641225618887,
                                ],
                            },
                            "coordinates_relative_polar": {
                                "event000": [
                                    [
                                        288.6270243358305,
                                        89.19086803726604,
                                        5.356313108184676,
                                    ]
                                ],
                                "event001": [
                                    [
                                        265.431817056372,
                                        86.6102430887874,
                                        6.574176515270801,
                                    ]
                                ],
                                "event002": [0.0, 0.0, 0.0],
                                "mic000": [
                                    [
                                        291.24062250983945,
                                        79.16583571951304,
                                        2.317769212833307,
                                    ]
                                ],
                            },
                        }
                    ],
                },
                "microphones": {
                    "mic000": {
                        "name": "ambeovr",
                        "micarray_type": "AmbeoVR",
                        "is_spherical": True,
                        "n_capsules": 4,
                        "capsule_names": ["FLU", "FRD", "BLD", "BRU"],
                        "coordinates_absolute": [
                            [4.030026609667739, -2.909095809315985, 1.6939148705637834],
                            [
                                4.030026609667739,
                                -2.9206803686227762,
                                1.6824433418367624,
                            ],
                            [4.018442050360947, -2.909095809315985, 1.6824433418367624],
                            [
                                4.018442050360947,
                                -2.9206803686227762,
                                1.6939148705637834,
                            ],
                        ],
                        "coordinates_polar": [
                            [45.0, 55.0, 0.01],
                            [315.0, 125.0, 0.01],
                            [135.0, 125.0, 0.01],
                            [225.0, 55.0, 0.01],
                        ],
                        "coordinates_center": [
                            4.024234330014343,
                            -2.9148880889693807,
                            1.688179106200273,
                        ],
                        "coordinates_cartesian": [
                            [
                                0.005792279653395693,
                                0.005792279653395692,
                                0.005735764363510461,
                            ],
                            [
                                0.00579227965339569,
                                -0.005792279653395693,
                                -0.005735764363510461,
                            ],
                            [
                                -0.005792279653395691,
                                0.005792279653395692,
                                -0.005735764363510461,
                            ],
                            [
                                -0.0057922796533956935,
                                -0.005792279653395692,
                                0.005735764363510461,
                            ],
                        ],
                    }
                },
                "mesh": {
                    "fname": "Oyens",
                    "ftype": ".glb",
                    "fpath": str(
                        utils.get_project_root()
                        / "tests/test_resources/meshes/Oyens.glb"
                    ),
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
        }
    ],
)
def test_scene_from_dict(input_dict: dict):
    ev = Scene.from_dict(input_dict)
    assert isinstance(ev, Scene)
    assert len(ev.events) == len(input_dict["events"])
    assert (
        len(ev.state.emitters)
        == len(input_dict["state"]["emitters"])
        == ev.state.ctx.get_source_count()
    )
    assert len(ev.ambience.keys()) == len(input_dict["ambience"])


@pytest.mark.parametrize(
    "filepath, emitter_kws, event_kws",
    [
        # Test 1: explicitly define a filepath, emitter keywords, and event keywords (overrides)
        (
            SOUNDEVENT_DIR / "music/000010.mp3",
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
