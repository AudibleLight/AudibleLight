#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for generating DCASE-like metadata"""

import numpy as np
import pytest

from audiblelight.core import Scene
from audiblelight.synthesize import generate_dcase2024_metadata
from tests import utils_tests


@pytest.mark.parametrize("duration", [20, 30])
def test_generate_dcase_2024_metadata_overlap(duration):
    """
    Test DCASE metadata creation with overlapping events
    """
    # Create a scene, add two music objects that overlap
    scene = Scene(duration=duration, mesh_path=utils_tests.OYENS_PATH)
    scene.add_microphone(microphone_type="ambeovr")
    scene.add_event(
        event_type="static",
        filepath=utils_tests.TEST_MUSICS[0],
        duration=10.0,
        scene_start=1.0,
    )
    scene.add_event(
        event_type="static",
        filepath=utils_tests.TEST_MUSICS[0],
        duration=10.0,
        scene_start=5.0,
    )

    # Generate the dcase metadata
    dcase = generate_dcase2024_metadata(scene)["mic000"]
    dcase = dcase.reset_index(drop=False).to_numpy()

    # Frame index should ascend
    assert np.array_equal(dcase[:, 0], np.sort(dcase[:, 0]))

    # Should have some duplicates in the frame counter
    assert len(dcase[:, 0]) > len(np.unique(dcase[:, 0]))

    # First event starts at 1 second in, last event ends 15 seconds in
    assert dcase[0, 0] == 10
    assert dcase[-1, 0] == 150

    # Two static events, expect two azimuth/elevation/distance values
    assert len(np.unique(dcase[:, 3])) == 2
    assert len(np.unique(dcase[:, 4])) == 2
    assert len(np.unique(dcase[:, 5])) == 2


@pytest.mark.parametrize("duration", [20, 30])
def test_generate_dcase_metadata_static(duration: int):
    """
    Test DCASE metadata creation with a single static event
    """
    # Create a scene, add single music object
    scene = Scene(duration=duration, mesh_path=utils_tests.OYENS_PATH)
    scene.add_microphone(microphone_type="ambeovr")
    scene.add_event(
        event_type="static",
        filepath=utils_tests.TEST_MUSICS[0],
        duration=10.0,
        scene_start=2.5,
    )

    # Generate the dcase metadata
    dcase = generate_dcase2024_metadata(scene)["mic000"]
    dcase = dcase.reset_index(drop=False).to_numpy()

    # Frame index should ascend
    assert np.array_equal(dcase[:, 0], np.sort(dcase[:, 0]))

    # No overlapping events, should not have duplicates
    assert len(np.unique(dcase[:, 0])) == len(dcase[:, 0])

    # Event starts 2.5 seconds in, finishes 12.5 seconds in
    assert dcase[0, 0] == 25
    assert dcase[-1, 0] == 125

    # All values should be correct
    expected_az, expected_el, expected_dist = (
        scene.get_event(0).emitters[0].coordinates_relative_polar["mic000"][0]
    )
    for actual_az, actual_el, actual_dist in dcase[:, 3:]:
        assert actual_az == round(expected_az)
        assert actual_el == round(expected_el)
        assert actual_dist == round(expected_dist * 100)


@pytest.mark.parametrize("duration", [20, 30])
def test_generate_dcase_metadata_moving(duration: int):
    """
    Test DCASE metadata creation with a single moving event
    """
    # Create a scene, add two music objects (one static, one moving)
    scene = Scene(duration=duration, mesh_path=utils_tests.OYENS_PATH)
    scene.add_microphone(microphone_type="ambeovr")
    scene.add_event(
        event_type="moving",
        filepath=utils_tests.TEST_MUSICS[1],
        duration=5.0,
        spatial_velocity=1.0,
        spatial_resolution=2.5,
        scene_start=5.0,
    )

    # Generate the dcase metadata
    dcase = generate_dcase2024_metadata(scene)["mic000"]
    dcase = dcase.reset_index(drop=False).to_numpy()

    # Frame index should ascend
    assert np.array_equal(dcase[:, 0], np.sort(dcase[:, 0]))

    # Event starts 5 seconds in, finishes 10 seconds in
    assert dcase[0, 0] == 50
    assert dcase[-1, 0] == 100

    # No overlapping events, should not have duplicates
    assert len(np.unique(dcase[:, 0])) == len(dcase[:, 0])

    # Starting and ending positions should be correct
    for idx in [0, -1]:
        actual_az, actual_el, actual_dist = dcase[idx, 3:]
        expected_az, expected_el, expected_dist = (
            scene.get_event(0).emitters[idx].coordinates_relative_polar["mic000"][0]
        )
        assert actual_az == round(expected_az)
        assert actual_el == round(expected_el)
        assert actual_dist == round(expected_dist * 100)


@pytest.mark.parametrize("duration", [20, 30])
def test_generate_dcase_2024_metadata_static_and_moving(duration: int):
    """
    Test DCASE metadata creation with both static and moving events
    """
    # Create a scene, add two music objects (one static, one moving)
    scene = Scene(duration=duration, mesh_path=utils_tests.OYENS_PATH)
    scene.add_microphone(microphone_type="ambeovr")
    scene.add_event(
        event_type="static",
        filepath=utils_tests.TEST_MUSICS[0],
        duration=1.0,
        scene_start=1.0,
    )
    scene.add_event(
        event_type="moving",
        filepath=utils_tests.TEST_MUSICS[1],
        duration=5.0,
        spatial_velocity=1.0,
        spatial_resolution=2.5,
        scene_start=5.0,
    )

    # Generate the dcase metadata
    dcase_out = generate_dcase2024_metadata(scene)

    # Scene only has one listener, so we should only have one dataframe
    assert len(dcase_out) == 1
    dcase = dcase_out["mic000"]

    # Frame index should ascend
    assert np.array_equal(dcase.index, np.sort(dcase.index))

    # No overlapping events, so frame index should be unique
    assert len(dcase.index.unique()) == len(dcase.index)

    # Ending coordinate for moving event should be correct
    actual_az, actual_el = dcase.iloc[-1][["azimuth", "elevation"]].to_list()
    expected_az, expected_el, _ = (
        scene.get_event(1).emitters[-1].coordinates_relative_polar["mic000"][0]
    )
    assert actual_az == round(expected_az)
    assert actual_el == round(expected_el)

    # Should have two different unique class IDs (we have two music objects)
    #  But we should only have one active class index (only one class, == music)
    assert dcase["source_number_index"].nunique() == 2
    assert dcase["active_class_index"].nunique() == 1

    # Number of frames should be smaller than total duration of scene / dcase_resolution
    assert dcase.index.max() <= (scene.duration / 0.1)

    # Azimuth/elevation should be in expected format
    assert dcase["azimuth"].min() >= -180
    assert dcase["azimuth"].max() <= 180
    assert dcase["elevation"].min() >= -90
    assert dcase["elevation"].max() <= 90


def test_dcase_metadata_bad():
    # Create a scene, add mic and event
    scene = Scene(duration=10, mesh_path=utils_tests.OYENS_PATH)
    scene.add_microphone(microphone_type="ambeovr")
    scene.add_event(
        event_type="static",
        filepath=utils_tests.TEST_MUSICS[0],
        duration=1.0,
        scene_start=1.0,
    )

    # Altering one of the class indices: should lead to an error as we expect this to be an int
    scene.events["event000"].class_id = "asdf"
    with pytest.raises(ValueError):
        _ = generate_dcase2024_metadata(scene)


@pytest.mark.parametrize(
    "events,expected",
    [
        # From https://dcase.community/challenge2024/task-audio-and-audiovisual-sound-event-localization-and-detection-with-source-distance-estimation
        (
            [
                dict(
                    position=[-50, 30, 1.81],
                    scene_start=1.0,
                    duration=0.1,
                    filepath=utils_tests.SOUNDEVENT_DIR / "maleSpeech/93853.wav",
                    alias="speech1",
                ),
                dict(
                    position=[10, -20, 2.43],
                    scene_start=1.1,
                    duration=0.2,
                    filepath=utils_tests.SOUNDEVENT_DIR / "maleSpeech/93856.wav",
                    alias="speech2",
                ),
                dict(
                    position=[-40, 0, 0.80],
                    scene_start=1.3,
                    duration=0.04,
                    filepath=utils_tests.TEST_MUSICS[0],
                    alias="music1",
                ),
            ],
            np.array(
                [
                    [10, 1, 0, -50, 30, 181],
                    [11, 1, 0, -50, 30, 181],
                    [11, 1, 1, 10, -20, 243],
                    [12, 1, 1, 10, -20, 243],
                    [13, 1, 1, 10, -20, 243],
                    [13, 8, 0, -40, 0, 80],
                ]
            ),
        ),
        # From dev-train-dcase/fold1_room1_mix001
        # Note that distance values/source IDs have been invented
        # Distance values are not provided in this dataset
        # Source IDs are formatted differently in AudibleLight (but they don't matter anyway)
        (
            [
                dict(
                    position=[95.0, 5.0, 1.0],
                    scene_start=10.0,
                    duration=0.5,
                    filepath=utils_tests.SOUNDEVENT_DIR / "musicInstrument/3471.wav",
                ),
                dict(
                    position=[129, -18, 0.5],
                    scene_start=10.2,
                    duration=0.3,
                    filepath=utils_tests.SOUNDEVENT_DIR / "laughter/9547.wav",
                ),
            ],
            np.array(
                [
                    [100, 9, 0, 95, 5, 100],
                    [101, 9, 0, 95, 5, 100],
                    [102, 4, 0, 129, -18, 50],
                    [102, 9, 0, 95, 5, 100],
                    [103, 4, 0, 129, -18, 50],
                    [103, 9, 0, 95, 5, 100],
                    [104, 4, 0, 129, -18, 50],
                    [104, 9, 0, 95, 5, 100],
                    [105, 4, 0, 129, -18, 50],
                    [105, 9, 0, 95, 5, 100],
                ]
            ),
        ),
        # From dev-train-tau/fold3_room4_mix001
        (
            [
                dict(
                    position=[-55.0, 9.0, 2.64],
                    scene_start=25.5,
                    duration=0.4,
                    filepath=utils_tests.SOUNDEVENT_DIR / "doorCupboard/35632.wav",
                ),
                dict(
                    position=[-61.0, -6.0, 2.18],
                    scene_start=27.5,
                    duration=0.5,
                    filepath=utils_tests.SOUNDEVENT_DIR / "waterTap/95709.wav",
                ),
            ],
            np.array(
                [
                    [255, 7, 0, -55, 9, 264],
                    [256, 7, 0, -55, 9, 264],
                    [257, 7, 0, -55, 9, 264],
                    [258, 7, 0, -55, 9, 264],
                    [259, 7, 0, -55, 9, 264],
                    [275, 10, 0, -61, -6, 218],
                    [276, 10, 0, -61, -6, 218],
                    [277, 10, 0, -61, -6, 218],
                    [278, 10, 0, -61, -6, 218],
                    [279, 10, 0, -61, -6, 218],
                    [280, 10, 0, -61, -6, 218],
                ]
            ),
        ),
    ],
)
def test_generate_dcase_2024_metadata_vs_example(events, expected):
    """
    Test DCASE metadata format versus a known example
    """
    # Create a Scene: can be any mesh here, we don't care
    example_scene = Scene(
        duration=30,
        mesh_path=utils_tests.OYENS_PATH,
        state_kwargs=dict(
            empty_space_around_surface=0.0,
        ),
    )

    # Add in a microphone at a specific, open position
    example_scene.add_microphone(
        microphone_type="ambeovr", position=[2.0, -2.5, 1.2], alias="poltest"
    )

    # Iterate through all the events and add
    for ev in events:
        created = example_scene.add_event(
            event_type="static", mic="poltest", polar=True, **ev
        )
        # Check our metadata format has parsed polar positions correctly
        created_position = created.get_emitter(0).coordinates_relative_polar["poltest"][
            0
        ]
        assert np.allclose(created_position, ev["position"])

    # Generate the metadata
    dcase_out = generate_dcase2024_metadata(example_scene)
    actual_out = dcase_out["poltest"].reset_index(drop=False).to_numpy()

    # Compare against the expected format
    assert np.allclose(actual_out, expected)
