#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/synthesize.py"""


import os
import subprocess
from time import time
from unittest.mock import PropertyMock, patch

import cv2
import librosa.util
import numpy as np
import pytest

import audiblelight.synthesize as syn
from audiblelight.event import Event
from audiblelight.worldstate import Emitter
from tests import utils_tests


@pytest.mark.parametrize(
    "audio, ir, error_message",
    [
        (
            np.ones((5, 2)),
            np.ones((5, 4)),
            "Only mono input is supported",
        ),  # audio not mono
        (np.ones(5), np.ones(5), "Expected shape of IR should be"),  # IR not 2D
        (np.ones(5), np.ones((5, 1, 1)), "Expected shape of IR should be"),  # IR not 2D
    ],
)
def test_time_invariant_convolution_invalid(audio, ir, error_message):
    with pytest.raises(ValueError, match=error_message):
        syn.time_invariant_convolution(audio, ir)


@pytest.mark.parametrize(
    "x, snr, expected_max",
    [
        (np.array([0.0, 0.5, -0.5, 1.0, -1.0]), 2.0, 2.0),
        (np.array([0.0, 0.25, -0.25]), 1.0, 1.0),
        (np.array([1.0, 2.0, 3.0]), 6.0, 6.0),
        (np.array([-1e-10, 1e-10]), 0.5, 0.5),
        (np.zeros(5), 3.0, 0.0),  # special case: zero input
    ],
)
def test_apply_snr(x, snr, expected_max):
    result = syn.apply_snr(x, snr)
    actual_max = np.max(np.abs(result))
    assert np.isclose(
        actual_max, expected_max, rtol=1e-5
    ), f"Expected max {expected_max}, got {actual_max}"


@pytest.mark.parametrize("n_emitters", list(range(4)))
def test_render_event_audio(n_emitters, oyens_scene_no_overlap):
    # Create the event
    ev = Event(alias="tester", filepath=utils_tests.TEST_AUDIOS[0], snr=5)
    # This is not the proper way to do this, but whatever
    emitter_list = []
    for emitter_idx in range(n_emitters):
        coords = oyens_scene_no_overlap.state.get_random_point_inside_mesh()
        em = Emitter(alias=f"emitter_{emitter_idx}", coordinates_absolute=coords)
        emitter_list.append(em)
    # Update the state and register the emitters to the event
    oyens_scene_no_overlap.state._update()
    if len(emitter_list) > 0:
        ev.register_emitters(emitter_list)
    else:
        ev.emitters = emitter_list
    # Create some dummy IRs
    irs = np.random.rand(4, n_emitters, 10000)
    # Do the generation
    syn.render_event_audio(
        ev, irs, mic_alias="mic000", ref_db=oyens_scene_no_overlap.ref_db
    )
    # Check everything
    assert hasattr(ev, "spatial_audio")
    assert isinstance(ev.spatial_audio, dict)
    assert isinstance(ev.spatial_audio["mic000"], np.ndarray)
    assert ev.spatial_audio["mic000"].shape[0] == 4
    assert ev.spatial_audio["mic000"].ndim == 2


@pytest.mark.parametrize(
    "n_events",
    [
        # 1,
        2,
    ],
)
def test_render_scene_audio_from_static_events(n_events: int, oyens_scene_no_overlap):
    oyens_scene_no_overlap.clear_events()
    # Add static sources in
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(event_type="static")

    syn.validate_scene(oyens_scene_no_overlap)
    init_time = time()
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap, ignore_cache=True)
    no_cache_time = time() - init_time
    assert len(oyens_scene_no_overlap.events) == n_events

    for event_alias, event in oyens_scene_no_overlap.events.items():
        assert isinstance(event.spatial_audio["mic000"], np.ndarray)
        n_channels, n_samples = event.spatial_audio["mic000"].shape
        # Number of channels should be same as microphone, number of samples should be same as audio
        assert n_channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
        assert n_samples == event.audio.shape[-1]

    # We should just be able to grab the audio from the cache now
    init_time = time()
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap, ignore_cache=False)
    cache_time = time() - init_time

    # With caching, should be much quicker to render the audio
    assert cache_time < no_cache_time

    # State should have IR objects cached
    assert isinstance(oyens_scene_no_overlap.state.irs, dict)


@pytest.mark.parametrize(
    "n_events",
    [
        1,
        2,
    ],
)
def test_render_scene_audio_from_moving_events(n_events: int, oyens_scene_no_overlap):
    oyens_scene_no_overlap.clear_events()
    # Add static sources in
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(
            filepath=utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            # Use predefined kwargs so rendering doesn't take ages
            event_type="moving",
            spatial_resolution=2,
            duration=1,
            spatial_velocity=1,
        )

    syn.validate_scene(oyens_scene_no_overlap)
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap)
    assert len(oyens_scene_no_overlap.events) == n_events

    for event_alias, event in oyens_scene_no_overlap.events.items():
        assert event.is_moving
        assert isinstance(event.spatial_audio["mic000"], np.ndarray)
        n_channels, n_samples = event.spatial_audio["mic000"].shape
        # Number of channels should be same as microphone, number of samples should be same as audio
        assert n_channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
        assert n_samples == event.load_audio().shape[-1]


@pytest.mark.parametrize(
    "n_events",
    [
        1,
        2,
    ],
)
def test_generate_scene_audio_from_events(n_events: int, oyens_scene_no_overlap):
    # Clear everything out for safety
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.clear_ambience()

    # Add both N static and N moving events
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(
            event_type="static",
        )
        # Predefined kwargs so rendering doesn't take ages
        oyens_scene_no_overlap.add_event(
            event_type="moving",
            spatial_resolution=2,
            duration=1,
            spatial_velocity=1,
        )

    # Add some ambience: white noise, mono audio, multichannel audio
    oyens_scene_no_overlap.add_ambience(noise="white")
    oyens_scene_no_overlap.add_ambience(
        filepath=utils_tests.SOUNDEVENT_DIR / "waterTap/240693.wav"
    )
    oyens_scene_no_overlap.add_ambience(
        filepath=utils_tests.TEST_RESOURCES
        / "spatialsoundevents/voice_whitenoise_foa.wav",
    )

    # Render the scene audio
    syn.validate_scene(oyens_scene_no_overlap)
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap)

    # Now, try generating the full scene audio
    syn.generate_scene_audio_from_events(oyens_scene_no_overlap)
    assert isinstance(oyens_scene_no_overlap.audio["mic000"], np.ndarray)

    # Audio should have the expected number of channels and duration
    channels, duration = oyens_scene_no_overlap.audio["mic000"].shape
    assert channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
    expected = round(
        oyens_scene_no_overlap.sample_rate * oyens_scene_no_overlap.duration
    )
    assert duration == expected

    # Also, check the "unpadded" Event audio
    for ev in oyens_scene_no_overlap.get_events():
        assert hasattr(ev, "_spatial_audio_padded")
        assert isinstance(ev._spatial_audio_padded, dict)
        assert isinstance(ev._spatial_audio_padded["mic000"], np.ndarray)

        # Should be identical to the Scene itself
        channels1, duration1 = ev._spatial_audio_padded["mic000"].shape
        assert channels1 == channels
        assert duration1 == duration

        # Shouldn't have dry + dry/padded audio
        for attr_ in ["_spatial_audio_dry", "_spatial_audio_dry_padded"]:
            assert hasattr(ev, attr_)
            assert len(getattr(ev, attr_)) == 0


def test_validate_scene(oyens_scene_factory):
    # Test with mismatching number of emitters/events/sources: need to mock
    scn = oyens_scene_factory()
    scn.add_event(event_type="static")
    with patch.object(
        type(scn.state), "num_emitters", new_callable=PropertyMock
    ) as mock_emitters:
        mock_emitters.return_value = 5
        with pytest.raises(ValueError, match="Mismatching number of emitters"):
            syn.validate_scene(scn)

    # Test with no ray-tracing engine sources
    scn = oyens_scene_factory()
    scn.add_event(event_type="static")
    scn.state.ctx.clear_sources()
    with pytest.raises(ValueError, match="Ray-tracing engine has no sources!"):
        syn.validate_scene(scn)

    # Test with no ray-tracing engine listeners
    scn.state.ctx.clear_listeners()
    with pytest.raises(ValueError, match="Ray-tracing engine has no listeners!"):
        syn.validate_scene(scn)

    # Test with no emitters
    scn = oyens_scene_factory()
    scn.clear_emitters()  # 1 microphone, 0 emitters, 0 events
    with pytest.raises(ValueError, match="WorldState has no emitters!"):
        syn.validate_scene(scn)

    # Test with no mics
    scn = oyens_scene_factory()
    scn.add_event(event_type="static")
    scn.clear_microphones()  # 1 emitter, 1 event, 0 microphones
    with pytest.raises(ValueError, match="WorldState has no microphones!"):
        syn.validate_scene(scn)

    # Test with no events
    scn = oyens_scene_factory()
    scn.state.add_emitter()  # 1 emitter, 1 mic, 0 events
    with pytest.raises(ValueError, match="Scene has no events!"):
        syn.validate_scene(scn)

    # Do the same for the capsules
    class TempMic:
        @property
        def n_listeners(self):
            return 5

    scn = oyens_scene_factory()
    scn.add_event(event_type="static")
    scn.state.microphones["asdf"] = TempMic()
    with pytest.raises(ValueError, match="Mismatching number of microphones"):
        syn.validate_scene(scn)

    # Test with events with no registered emitters
    scn = oyens_scene_factory()
    out = scn.add_event(event_type="static")
    out.emitters = None
    with pytest.raises(ValueError, match="Event with alias"):
        syn.validate_scene(scn)


def test_validate_sofa_scene(metu_scene_factory):
    # Test invalid
    scn = metu_scene_factory()
    scn.state.add_emitter()  # 1 emitter, 1 mic, 0 events
    with pytest.raises(ValueError, match="Scene has no events!"):
        syn.validate_scene(scn)

    # Test valid
    scn = metu_scene_factory()
    scn.add_event(alias="temp")
    try:
        syn.validate_scene(scn)
    except ValueError:
        pytest.fail()


@pytest.mark.parametrize(
    "db, x, expected_multiplier",
    [
        (0, 1.0, 1.0),
        (6.0206, 1.0, 2.0),
        (-6.0206, 1.0, 0.5),
        (20.0, 0.1, 100.0),
        (-20.0, 10.0, 0.01),
    ],
)
def test_db_to_multiplier(db, x, expected_multiplier):
    result = syn.db_to_multiplier(db, x)

    # we expect a small but finite multiplier, not exactly 0
    assert np.isfinite(result), "Result should be finite even when x is 0"
    assert result > 0, "Multiplier should be positive"
    assert not np.isnan(result), "Result should not be NaN"
    assert not np.isinf(result), "Result should not be Inf"

    # should be near expected value
    assert np.isclose(result, expected_multiplier, atol=1e-4)

    # now multiply by the scalar
    audio = np.random.rand(1000)

    # should be valid after scaling
    scaled = audio * result
    try:
        librosa.util.valid_audio(scaled)
    except librosa.util.exceptions.ParameterError as e:
        pytest.fail(e)


@pytest.mark.parametrize("n_moving, n_static", [(1, 3), (2, 2)])
def test_normalize_irs(n_moving, n_static, oyens_scene_no_overlap):
    # Add some moving and static events
    for i in range(n_static):
        oyens_scene_no_overlap.add_event(event_type="static", duration=1.0)
    for i in range(n_moving):
        oyens_scene_no_overlap.add_event(
            event_type="moving", duration=5.0, spatial_resolution=1.0
        )

    # Simulate, get IRs for the microphone
    oyens_scene_no_overlap.state.simulate()
    irs = oyens_scene_no_overlap.state.get_irs()
    mic_ir = irs["mic000"]

    # We need a separate counter for each microphone
    emitter_counter = 0

    # Iterate over all events
    for event_alias, event in oyens_scene_no_overlap.events.items():

        # Grab the IRs for the current event's emitters and check (not normalized)
        event_irs = mic_ir[:, emitter_counter : len(event) + emitter_counter, :]
        energies = np.mean(np.sqrt(np.sum(np.power(np.abs(event_irs), 2), axis=-1)))
        assert not pytest.approx(energies) == 1.0

        # Normalize the IRs and check
        event_irs_norm = syn.normalize_irs(event_irs)
        energies = np.mean(
            np.sqrt(np.sum(np.power(np.abs(event_irs_norm), 2), axis=-1))
        )
        assert pytest.approx(energies) == 1.0

        # Shapes should be the same, but audio should not be
        assert np.array_equal(event_irs.shape, event_irs_norm.shape)
        assert not np.array_equal(event_irs, event_irs_norm)

        # Update the counter
        emitter_counter += len(event)


@pytest.mark.parametrize(
    "event_kwargs",
    [
        dict(ref_ir_channel=None, direct_path_time_ms=None),
        dict(ref_ir_channel=0, direct_path_time_ms=[5, 60]),
        dict(ref_ir_channel=0, direct_path_time_ms=None),
    ],
)
def test_compute_dry_audio(event_kwargs, oyens_scene_no_overlap):
    # Add static event with set kwargs
    oyens_scene_no_overlap.clear_events()
    oyens_scene_no_overlap.add_event(
        event_type="static",
        duration=5,
        event_start=5,
        scene_start=5,
        filepath=utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
        alias="dry",
        **event_kwargs,
    )

    # Do the synthesis
    syn.validate_scene(oyens_scene_no_overlap)
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap, ignore_cache=False)
    syn.generate_scene_audio_from_events(oyens_scene_no_overlap)

    # Grab the event
    event = oyens_scene_no_overlap.get_event("dry")
    assert event.is_audio_loaded

    # Check dry audio if required
    if event.ref_ir_channel is not None and event.direct_path_time_ms is not None:
        assert hasattr(event, "_spatial_audio_dry")
        assert event._spatial_audio_dry is not None

        # Audio should be 1D
        audio = event._spatial_audio_dry["mic000"]
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1

        # Padded audio should also be 1D
        padded_audio = event._spatial_audio_dry_padded["mic000"]
        assert isinstance(padded_audio, np.ndarray)
        assert padded_audio.ndim == 1

        # However, we expect more samples (it is padded)
        assert len(padded_audio) >= len(audio)

        # Min/max of the audio should be the same, however
        assert pytest.approx(padded_audio.min()) == audio.min()
        assert pytest.approx(padded_audio.max()) == audio.max()


@pytest.mark.parametrize(
    "audio_filepath,image_filepath",
    [
        (
            utils_tests.SOUNDEVENT_DIR / "telephone/30085.wav",
            utils_tests.IMAGE_DIR / "telephone/3_0.jpg",
        ),
        (
            utils_tests.SOUNDEVENT_DIR / "waterTap/95709.wav",
            utils_tests.IMAGE_DIR / "waterTap/32_0.jpg",
        ),
        (
            utils_tests.SOUNDEVENT_DIR / "maleSpeech/93853.wav",
            utils_tests.IMAGE_DIR / "maleSpeech/2_0.jpg",
        ),
    ],
)
def test_video_generation(audio_filepath, image_filepath, oyens_scene_with_images):
    # add some short events with a set audio file and image
    oyens_scene_with_images.add_event(
        event_type="static",
        filepath=audio_filepath,
        image_filepath=image_filepath,
        duration=0.5,
        scene_start=2.5,
        # add at a known position: definitely visible from microphone
        position=[1.5, -0.5, 0.5],
    )
    ev = oyens_scene_with_images.add_event(
        event_type="moving",
        shape="linear",
        filepath=audio_filepath,
        image_filepath=image_filepath,
        duration=0.5,
        scene_start=7.5,
        spatial_velocity=0.5,
        spatial_resolution=2.0,
        # start at another known position
        position=[2.5, -3.0, 1.0],
    )

    video_out_path = utils_tests.TEST_RESOURCES / f"video_test_{ev.class_label}"
    audio_out_path = utils_tests.TEST_RESOURCES / f"audio_test_{ev.class_label}"
    oyens_scene_with_images.generate(
        audio=True,
        metadata_dcase=False,
        metadata_json=False,
        video=True,
        video_fname=video_out_path,
        audio_fname=audio_out_path,
    )

    # Video file should exist
    video_out_path_full = video_out_path.with_name(
        f"video_test_{ev.class_label}_mic000.mp4"
    )
    assert video_out_path_full.exists()

    # Audio file should exist
    audio_out_path_full = audio_out_path.with_name(
        f"audio_test_{ev.class_label}_mic000.wav"
    )
    assert audio_out_path_full.exists()

    # Read video in openCV
    vid = cv2.VideoCapture(video_out_path_full)

    # Get properties of the video file
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Should be the same as set in the Scene object
    assert height == oyens_scene_with_images.video_res[1]
    assert width == oyens_scene_with_images.video_res[0]
    assert fps == oyens_scene_with_images.video_fps
    assert pytest.approx(duration) == oyens_scene_with_images.duration

    # Manual checks:
    #  1) Check video file manually to ensure that the two images appear correctly
    #  2) Combine with `ffmpeg -i video_test_mic000.mp4 -i audio_test_mic000.wav -c:v copy -c:a aac output.mp4`
    #       Check video + audio start and stop at same time
    output_video_path = utils_tests.TEST_RESOURCES / f"output_{ev.class_label}.mp4"
    subprocess.call(
        f"ffmpeg -i {str(video_out_path_full)} -i {str(audio_out_path_full)} -c:v libx264 -c:a aac -y {str(output_video_path)}",
        shell=True,
    )
    os.remove(video_out_path_full)
    os.remove(audio_out_path_full)
