#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define test cases that use algorithms (e.g., MUSIC) and basic acoustic principles to test AudibleLight outputs"""

import os

import numpy as np
import pytest
import soundfile as sf
from pyroomacoustics.doa.music import MUSIC
from scipy.signal import stft

from audiblelight import config
from audiblelight.augmentation import AudioChannelSwapping
from audiblelight.core import Scene
from audiblelight.synthesize import generate_dcase2024_metadata
from tests import utils_tests


def _apply_music(test_scene: Scene, **kwargs) -> MUSIC:
    # Coordinates of our capsules for the eigenmike
    l_: np.ndarray = test_scene.get_microphone("tester").coordinates_absolute
    l_ = l_.T

    # Get the parameters
    fs = int(test_scene.sample_rate)
    nfft = config.FFT_SIZE
    num_sources = len(test_scene.get_events())
    freq_range = [300, 3500]
    assert num_sources == len(test_scene.get_events())

    # Create the MUSIC object
    #  Ensure azimuth is in range [-180, 180], increasing counter-clockwise
    #  Ensure colatitude is in range [-90, 90], where 0 == straight ahead
    music = MUSIC(L=l_, fs=fs, nfft=nfft, num_sources=num_sources, **kwargs)

    # Compute the STFT
    stft_signals = stft(
        test_scene.audio["tester"], fs=fs, nperseg=nfft, noverlap=0, boundary=None
    )[2]
    x, y, _ = stft_signals.shape
    assert x == test_scene.get_microphone("tester").n_capsules
    assert y == (nfft / 2) + 1

    # Locate the sources
    music.locate_sources(stft_signals, num_src=num_sources, freq_range=freq_range)

    return music


@pytest.mark.parametrize(
    "microphone,events",
    [
        # mic now placed in bedroom 1
        (
            [-1.5, -1.5, 0.7],
            [[90, 0, 0.2]],
        ),
        (
            [-1.5, -1.5, 0.7],
            [[-90, 0, 0.2]],
        ),
        # mic now placed in bedroom 2
        (
            [2.9, -7.0, 0.3],
            [[0.0, 0, 0.2]],
        ),
        (
            [2.9, -7.0, 0.3],
            [[180, 0, 0.2]],
        ),
        # mic now placed in living room
        (
            [2.5, -1.0, 0.5],
            [[45, 0, 0.2]],
        ),
    ],
)
@pytest.mark.flaky(reruns=3)
def test_simulated_azimuth_with_music(microphone: list, events: list):
    """
    Tests azimuth of simulated sound events and microphones with MUSIC algorithm.

    Places an Eigenmike32, simulates sound emitters, runs MUSIC, checks that estimated azimuth is near to actual
    """
    # Create a simulated scene
    test_scene = Scene(
        duration=5.1,
        mesh_path=utils_tests.OYENS_PATH,
        fg_path=utils_tests.SOUNDEVENT_DIR / "music",
        allow_duplicate_audios=False,
        max_overlap=3,
        state_kwargs=dict(rlr_kwargs=dict(sample_rate=16000)),
    )

    # Add the microphone and events
    test_scene.add_microphone(
        microphone_type="eigenmike32",
        position=microphone,
        keep_existing=False,
        alias="tester",
    )
    for emit in events:
        test_scene.add_event(
            position=emit, event_type="static", duration=5, event_start=0, polar=True
        )

    # Generate the audio and grab
    test_scene.generate(metadata_json=False, metadata_dcase=False, audio=False)

    # Create the MUSIC object, azimuth only for now
    music = _apply_music(
        test_scene,
        azimuth=np.deg2rad(np.arange(-180, 180)),
        colatitude=np.deg2rad(np.arange(-90, 90)),
        dim=2,
    )

    # Check the azimuth vs actual values
    located_azimuth = np.sort(np.rad2deg(music.azimuth_recon))
    actual_azimuth = np.sort(np.array(events)[:, 0])
    assert len(located_azimuth) == len(test_scene.get_events())
    assert np.allclose(located_azimuth, actual_azimuth, atol=30)


@pytest.mark.parametrize(
    "closemic_position,farmic_position,emitter_position",
    [
        # Testing "length-wise" in the room
        (
            [1.0, -9.5, 0.7],
            [1.0, 0.5, 0.7],
            [1.0, -8.5, 0.7],
        ),
        # Testing "width-wise" in the room
        (
            [0.5, -3.5, 0.7],
            [5.5, -3.5, 0.7],
            [1.5, -3.5, 0.7],
        ),
        # Testing "vertical-wise" in the room
        (
            [0.5, -3.5, 0.3],
            [0.5, -3.5, 1.1],
            [0.5, -3.5, 0.5],
        ),
    ],
)
def test_simulated_sound_distance_vs_two_mics(
    closemic_position: list, farmic_position: list, emitter_position: list
):
    """
    Tests distance of simulated sound emitters and microphones.

    Places an event and two AmbeoVR microphones near and far, then checks that the sound hits the close mic before far
    """
    # Create a simulated scene
    test_scene = Scene(
        duration=5.1,
        mesh_path=utils_tests.OYENS_PATH,
        fg_path=utils_tests.SOUNDEVENT_DIR / "music",
        max_overlap=3,
        state_kwargs=dict(rlr_kwargs=dict(sample_rate=16000)),
    )

    # Add the microphone and events
    for mic_pos, mic_name in zip(
        [closemic_position, farmic_position], ["close", "far"]
    ):
        test_scene.add_microphone(
            microphone_type="ambeovr",
            position=mic_pos,
            keep_existing=True,
            alias=f"{mic_name}mic",
        )
    test_scene.add_event(
        event_type="static", position=emitter_position, duration=1.0, event_start=1.0
    )

    # Do the generation, grab the IRs
    test_scene.generate(audio=False, metadata_dcase=False, metadata_json=False)
    output = test_scene.state.get_irs()

    # Shape of the output should be as expected
    assert len(output.keys()) == 2

    # Get the IDX of the sample at which the sound hits both microphones
    arrival_close = min(np.flatnonzero(output["closemic"]))
    arrival_far = min(np.flatnonzero(output["farmic"]))

    # IRs and Audio should be different
    assert not np.array_equal(output["closemic"], output["farmic"])
    assert not np.array_equal(test_scene.audio["closemic"], test_scene.audio["farmic"])

    # Should hit the closer mic before the further mic
    assert arrival_close < arrival_far


@pytest.mark.parametrize(
    "closeevent_position,farevent_position,mic_position",
    [
        ([-90.0, 0.0, 1.0], [90.0, 0.0, 3.0], [2.0, -2.5, 1.2]),
        ([0.0, 45.0, 0.5], [0.0, -45.0, 1.0], [2.0, -2.5, 1.2]),
    ],
)
def test_simulated_sound_distance_vs_two_events(
    closeevent_position, farevent_position, mic_position
):
    """
    Tests distance of simulated sound emitters and microphones.

    Places two events near and far, then checks that the near sound hits the mic before the far sound
    """
    # Create a simulated scene
    test_scene = Scene(
        duration=5.1,
        mesh_path=utils_tests.OYENS_PATH,
        fg_path=utils_tests.SOUNDEVENT_DIR / "music",
        max_overlap=3,
        state_kwargs=dict(rlr_kwargs=dict(sample_rate=16000)),
    )

    # Add the microphone and events
    test_scene.add_microphone(
        microphone_type="ambeovr",
        position=mic_position,
    )
    for event_pos, event_name in zip(
        [closeevent_position, farevent_position], ["close", "far"]
    ):
        test_scene.add_event(
            event_type="static",
            position=event_pos,
            polar=True,
            alias=f"{event_name}event",
            duration=1.0,
            event_start=1.0,
        )

    # Do the generation, grab the IRs
    test_scene.generate(audio=False, metadata_dcase=False, metadata_json=False)
    output = test_scene.state.get_irs()

    # Separate IR by event
    closeevent_ir = output["mic000"][:, 0, :]
    farevent_ir = output["mic000"][:, 1, :]

    # IR audio should be different
    assert not np.array_equal(closeevent_ir, farevent_ir)

    # Get the IDX of the sample at which the sound hits both microphones
    arrival_close = min(np.flatnonzero(closeevent_ir))
    arrival_far = min(np.flatnonzero(farevent_ir))

    # Closer event should hit before further event
    assert arrival_close < arrival_far


@pytest.mark.parametrize(
    "microphone,event",
    [
        # mic now placed in bedroom 1
        (
            [-1.5, -1.5, 0.7],
            [90, 0, 0.2],
        ),
        # mic now placed in bedroom 2
        (
            [2.9, -7.0, 0.3],
            [180, 0, 0.2],
        ),
        # mic now placed in living room
        (
            [2.5, -1.0, 0.5],
            [90, 0, 0.2],
        ),
    ],
)
def test_channel_swapping_with_music(
    microphone: list, event: list, oyens_scene_no_overlap: Scene
):
    """
    Test augmentation.AudioChannelSwapping with MUSIC (synthesising some basic white noise)
    """
    # Create a simulated scene
    test_scene = Scene(
        duration=1.1,
        mesh_path=utils_tests.OYENS_PATH,
        fg_path=utils_tests.SOUNDEVENT_DIR / "music",
        allow_duplicate_audios=False,
        max_overlap=2,
        state_kwargs=dict(rlr_kwargs=dict(sample_rate=16000)),
    )

    # Add the microphone
    test_scene.add_microphone(
        microphone_type="ambeovr",
        position=microphone,
        keep_existing=False,
        alias="tester",
    )

    # Create 5 seconds of random noise, save to temp file, add as event
    white_noise = np.random.uniform(-1, 1, int(test_scene.sample_rate * 1))
    sf.write("tmp.wav", white_noise, int(test_scene.sample_rate))
    test_scene.add_event(
        event_type="static",
        filepath=utils_tests.TEST_MUSICS[0],
        position=event,
        duration=1.0,
        polar=True,
        event_start=0.0,
        class_id=0,
        class_label="tmp",
    )

    # Generate the audio and grab: shape (8, 4, samples)
    test_scene.generate(metadata_json=False, metadata_dcase=False, audio=False)
    aout = test_scene.audio["tester"]
    mout = generate_dcase2024_metadata(test_scene)

    # Swap the labels and audio
    swapper = AudioChannelSwapping(
        test_scene.sample_rate,
        input_format=test_scene.get_microphone("tester").channel_layout,
    )
    swapped_audio = swapper(aout)
    swapped_labels = swapper.swap_labels(
        mout["tester"].reset_index(drop=False).to_numpy()
    )

    for swapped_a, swapped_l in zip(swapped_audio, swapped_labels):
        # Coordinates of our capsules
        l_: np.ndarray = test_scene.get_microphone("tester").coordinates_absolute
        l_ = l_.T

        # Get the parameters
        fs = int(test_scene.sample_rate)
        nfft = 2048
        num_sources = len(test_scene.get_events())

        # Create the MUSIC object
        #  Ensure azimuth is in range [-180, 180], increasing counter-clockwise
        #  Ensure colatitude is in range [-90, 90], where 0 == straight ahead
        music = MUSIC(
            L=l_,
            fs=fs,
            azimuth=np.deg2rad(np.arange(-180, 181)),
            colatitude=np.deg2rad(np.arange(-90, 91)),
            nfft=nfft,
            num_sources=num_sources,
            dim=2,
        )

        # Compute the STFT
        stft_signals = stft(swapped_a, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[
            2
        ]
        x, y, _ = stft_signals.shape
        assert x == test_scene.get_microphone("tester").n_capsules
        assert y == (nfft / 2) + 1

        # Locate the sources
        music.locate_sources(stft_signals, num_src=num_sources)
        srce = np.rad2deg(music.azimuth_recon)[0]
        fmt = ((srce + 180) % 360) - 180
        print(fmt, swapped_l[0, 3])

    os.remove("tmp.wav")
