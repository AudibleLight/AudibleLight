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
TEST_AUDIOS = sorted(
    [
        os.path.join(xs, x)
        for xs in TEST_AUDIO_DIRS
        for x in os.listdir(xs)
        if x.endswith((".wav", ".mp3"))
    ]
)


@pytest.mark.parametrize("audio_fpath", TEST_AUDIOS[:5])
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
        (TEST_AUDIOS[6], 0.5, 0.5),
        (TEST_AUDIOS[7], 1.0, 1.0),
        (TEST_AUDIOS[8], None, 1.0),
        (TEST_AUDIOS[6], 0.5, None),
        (TEST_AUDIOS[7], None, None),
        (TEST_AUDIOS[8], 2.0, 5.0),
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
        filepath=SOUNDEVENT_DIR
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
                        [-2.2333563194364796, -1.4665946631472622, 0.3130778986929861]
                    ],
                    "event002": [
                        [-1.7106695658740882, 5.0752318593323515, -0.07563942967148884]
                    ],
                },
                "relative_polar": {
                    "event000": [0.0, 0.0, 0.0],
                    "mic000": [
                        [106.69774947852416, 83.3402562971731, 3.104386347271515]
                    ],
                    "event001": [
                        [213.29200206063442, 83.31675945948695, 2.6901297601024576]
                    ],
                    "event002": [
                        [108.62702433583053, 90.80913196273397, 5.356313108184676]
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
                        [-2.2333563194364796, -1.4665946631472622, 0.3130778986929861]
                    ],
                    "event002": [
                        [-1.7106695658740882, 5.0752318593323515, -0.07563942967148884]
                    ],
                },
                "relative_polar": {
                    "event000": [0.0, 0.0, 0.0],
                    "mic000": [
                        [106.69774947852416, 83.3402562971731, 3.104386347271515]
                    ],
                    "event001": [
                        [213.29200206063442, 83.31675945948695, 2.6901297601024576]
                    ],
                    "event002": [
                        [108.62702433583053, 90.80913196273397, 5.356313108184676]
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
                            [106.69774947852416, 83.3402562971731, 3.104386347271515]
                        ],
                        "event001": [
                            [213.29200206063442, 83.31675945948695, 2.6901297601024576]
                        ],
                        "event002": [
                            [108.62702433583053, 90.80913196273397, 5.356313108184676]
                        ],
                    },
                }
            ],
        },
        {
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
                        [1.7106695658740882, -5.0752318593323515, 0.07563942967148884]
                    ],
                    "event001": [
                        [-0.5226867535623914, -6.541826522479614, 0.3887173283644749]
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
                        [288.6270243358305, 89.19086803726604, 5.356313108184676]
                    ],
                    "event001": [
                        [265.431817056372, 86.6102430887874, 6.574176515270801]
                    ],
                    "event002": [0.0, 0.0, 0.0],
                    "mic000": [
                        [291.24062250983945, 79.16583571951304, 2.317769212833307]
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
                        [1.7106695658740882, -5.0752318593323515, 0.07563942967148884]
                    ],
                    "event001": [
                        [-0.5226867535623914, -6.541826522479614, 0.3887173283644749]
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
                        [288.6270243358305, 89.19086803726604, 5.356313108184676]
                    ],
                    "event001": [
                        [265.431817056372, 86.6102430887874, 6.574176515270801]
                    ],
                    "event002": [0.0, 0.0, 0.0],
                    "mic000": [
                        [291.24062250983945, 79.16583571951304, 2.317769212833307]
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
                            [288.6270243358305, 89.19086803726604, 5.356313108184676]
                        ],
                        "event001": [
                            [265.431817056372, 86.6102430887874, 6.574176515270801]
                        ],
                        "event002": [0.0, 0.0, 0.0],
                        "mic000": [
                            [291.24062250983945, 79.16583571951304, 2.317769212833307]
                        ],
                    },
                }
            ],
        },
        {
            "alias": "event000",
            "filename": "205695.wav",
            "filepath": str(
                utils.get_project_root()
                / "tests/test_resources/soundevents/waterTap/205695.wav"
            ),
            "class_id": None,
            "class_label": None,
            "scene_start": 11.689693873802927,
            "scene_end": 17.75010203706823,
            "event_start": 0.0,
            "event_end": 6.060408163265306,
            "duration": 6.060408163265306,
            "snr": 3.3177864922712916,
            "sample_rate": 44100.0,
            "spatial_resolution": 3.960075785054248,
            "spatial_velocity": 0.7069086045527073,
            "start_coordinates": {
                "absolute": [
                    1.9158415874324373,
                    0.6803947853283612,
                    1.8840126269745547,
                ],
                "relative_cartesian": {
                    "event000": [0.0, 0.0, 0.0],
                    "mic000": [
                        -4.253492521453725,
                        0.2707495088936769,
                        0.5421344176136262,
                    ],
                    "event001": [
                        [0.024228355026296722, 0.5117105399471034, 0.5461066304932936]
                    ],
                    "event002": [
                        [-0.7403866829829435, -1.687673917159382, 0.16026281134720355]
                    ],
                    "event003": [
                        [-0.3842250283645088, 0.38109414846899514, 0.05925029661084702]
                    ],
                    "event004": [
                        [-5.58351584024455, -2.410336768711995, 0.070139295314543]
                    ],
                    "event005": [
                        [-0.7074928252981443, 0.036009991847026246, 0.28708787648205925]
                    ],
                    "event006": [
                        [-0.6736062870988979, 0.3607101182407948, 1.5702352442032335]
                    ],
                    "event007": [
                        [-1.1287840599288743, -2.130180914274869, 0.6744484896286989]
                    ],
                    "event008": [
                        [-4.048549197119175, 0.6778605283956853, 0.7829351948843912]
                    ],
                },
                "relative_polar": {
                    "event000": [0.0, 0.0, 0.0],
                    "mic000": [
                        [176.35783924078564, 82.75096966026958, 4.296441976029724]
                    ],
                    "event001": [
                        [87.28919687120414, 43.16962984107087, 0.7487770975056429]
                    ],
                    "event002": [
                        [246.31282090763125, 85.03002952148852, 1.8498918508116866]
                    ],
                    "event003": [
                        [135.23439266764623, 83.75179098203324, 0.5444007899218088]
                    ],
                    "event004": [
                        [203.3492777898867, 89.33923082053467, 6.081964485066811]
                    ],
                    "event005": [
                        [177.08627200536225, 67.93938193241128, 0.7643705032143453]
                    ],
                    "event006": [
                        [151.83135039683188, 25.948410971319564, 1.7462806021823754]
                    ],
                    "event007": [
                        [242.080858494601, 74.37026829854038, 2.5033387598670824]
                    ],
                    "event008": [
                        [170.49496501777205, 79.20158009311552, 4.1789033270512315]
                    ],
                },
            },
            "end_coordinates": {
                "absolute": [
                    1.9158415874324373,
                    0.6803947853283612,
                    1.8840126269745547,
                ],
                "relative_cartesian": {
                    "event000": [0.0, 0.0, 0.0],
                    "mic000": [
                        -4.253492521453725,
                        0.2707495088936769,
                        0.5421344176136262,
                    ],
                    "event001": [
                        [0.024228355026296722, 0.5117105399471034, 0.5461066304932936]
                    ],
                    "event002": [
                        [-0.7403866829829435, -1.687673917159382, 0.16026281134720355]
                    ],
                    "event003": [
                        [-0.3842250283645088, 0.38109414846899514, 0.05925029661084702]
                    ],
                    "event004": [
                        [-5.58351584024455, -2.410336768711995, 0.070139295314543]
                    ],
                    "event005": [
                        [-0.7074928252981443, 0.036009991847026246, 0.28708787648205925]
                    ],
                    "event006": [
                        [-0.6736062870988979, 0.3607101182407948, 1.5702352442032335]
                    ],
                    "event007": [
                        [-1.1287840599288743, -2.130180914274869, 0.6744484896286989]
                    ],
                    "event008": [
                        [-4.048549197119175, 0.6778605283956853, 0.7829351948843912]
                    ],
                },
                "relative_polar": {
                    "event000": [0.0, 0.0, 0.0],
                    "mic000": [
                        [176.35783924078564, 82.75096966026958, 4.296441976029724]
                    ],
                    "event001": [
                        [87.28919687120414, 43.16962984107087, 0.7487770975056429]
                    ],
                    "event002": [
                        [246.31282090763125, 85.03002952148852, 1.8498918508116866]
                    ],
                    "event003": [
                        [135.23439266764623, 83.75179098203324, 0.5444007899218088]
                    ],
                    "event004": [
                        [203.3492777898867, 89.33923082053467, 6.081964485066811]
                    ],
                    "event005": [
                        [177.08627200536225, 67.93938193241128, 0.7643705032143453]
                    ],
                    "event006": [
                        [151.83135039683188, 25.948410971319564, 1.7462806021823754]
                    ],
                    "event007": [
                        [242.080858494601, 74.37026829854038, 2.5033387598670824]
                    ],
                    "event008": [
                        [170.49496501777205, 79.20158009311552, 4.1789033270512315]
                    ],
                },
            },
            "emitters": [
                {
                    "alias": "event000",
                    "coordinates_absolute": [
                        1.9158415874324373,
                        0.6803947853283612,
                        1.8840126269745547,
                    ],
                    "coordinates_relative_cartesian": {
                        "event000": [0.0, 0.0, 0.0],
                        "mic000": [
                            -4.253492521453725,
                            0.2707495088936769,
                            0.5421344176136262,
                        ],
                        "event001": [
                            [
                                0.024228355026296722,
                                0.5117105399471034,
                                0.5461066304932936,
                            ]
                        ],
                        "event002": [
                            [
                                -0.7403866829829435,
                                -1.687673917159382,
                                0.16026281134720355,
                            ]
                        ],
                        "event003": [
                            [
                                -0.3842250283645088,
                                0.38109414846899514,
                                0.05925029661084702,
                            ]
                        ],
                        "event004": [
                            [-5.58351584024455, -2.410336768711995, 0.070139295314543]
                        ],
                        "event005": [
                            [
                                -0.7074928252981443,
                                0.036009991847026246,
                                0.28708787648205925,
                            ]
                        ],
                        "event006": [
                            [
                                -0.6736062870988979,
                                0.3607101182407948,
                                1.5702352442032335,
                            ]
                        ],
                        "event007": [
                            [
                                -1.1287840599288743,
                                -2.130180914274869,
                                0.6744484896286989,
                            ]
                        ],
                        "event008": [
                            [-4.048549197119175, 0.6778605283956853, 0.7829351948843912]
                        ],
                    },
                    "coordinates_relative_polar": {
                        "event000": [0.0, 0.0, 0.0],
                        "mic000": [
                            [176.35783924078564, 82.75096966026958, 4.296441976029724]
                        ],
                        "event001": [
                            [87.28919687120414, 43.16962984107087, 0.7487770975056429]
                        ],
                        "event002": [
                            [246.31282090763125, 85.03002952148852, 1.8498918508116866]
                        ],
                        "event003": [
                            [135.23439266764623, 83.75179098203324, 0.5444007899218088]
                        ],
                        "event004": [
                            [203.3492777898867, 89.33923082053467, 6.081964485066811]
                        ],
                        "event005": [
                            [177.08627200536225, 67.93938193241128, 0.7643705032143453]
                        ],
                        "event006": [
                            [151.83135039683188, 25.948410971319564, 1.7462806021823754]
                        ],
                        "event007": [
                            [242.080858494601, 74.37026829854038, 2.5033387598670824]
                        ],
                        "event008": [
                            [170.49496501777205, 79.20158009311552, 4.1789033270512315]
                        ],
                    },
                }
            ],
        },
    ],
)
def test_event_from_dict(input_dict: dict):
    ev = Event.from_dict(input_dict)
    assert isinstance(ev, Event)
    out_dict = ev.to_dict()
    for k, v in out_dict.items():
        assert input_dict[k] == out_dict[k]


@pytest.mark.parametrize("audio_fpath", TEST_AUDIOS[:5])
def test_magic_methods(audio_fpath: str, oyens_space):
    oyens_space.add_emitter(alias="test_emitter")
    emitter = oyens_space.get_emitters("test_emitter")
    ev = Event(audio_fpath, "test_event", emitters=emitter)
    # Iterate over all the magic methods that return strings
    for att in ["__str__", "__repr__"]:
        assert isinstance(getattr(ev, att)(), str)
    # Check the __eq__ comparison for identical objects
    assert ev == Event.from_dict(ev.to_dict())
