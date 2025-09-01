#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/synthesize.py"""


from time import time

import librosa.util
import numpy as np
import pytest

import audiblelight.synthesize as syn
from audiblelight.core import Scene
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
    syn.render_event_audio(ev, irs, oyens_scene_no_overlap.ref_db)
    # Check everything
    assert hasattr(ev, "spatial_audio")
    assert isinstance(ev.spatial_audio, np.ndarray)
    assert ev.spatial_audio.shape[0] == 4
    assert ev.spatial_audio.ndim == 2


@pytest.mark.parametrize(
    "n_events",
    [
        1,
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
        assert isinstance(event.spatial_audio, np.ndarray)
        n_channels, n_samples = event.spatial_audio.shape
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
        assert isinstance(event.spatial_audio, np.ndarray)
        n_channels, n_samples = event.spatial_audio.shape
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
    assert isinstance(oyens_scene_no_overlap.audio, np.ndarray)

    # Audio should have the expected number of channels and duration
    channels, duration = oyens_scene_no_overlap.audio.shape
    assert channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
    expected = round(
        oyens_scene_no_overlap.state.ctx.config.sample_rate
        * oyens_scene_no_overlap.duration
    )
    assert duration == expected


def test_validate_scene(oyens_scene_factory):
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

    # Test with no ray-tracing listeners
    scn = oyens_scene_factory()
    scn.state.add_emitter()
    scn.state.ctx.clear_listeners()
    with pytest.raises(ValueError, match="Ray-tracing engine has no listeners!"):
        syn.validate_scene(scn)

    # Test with no ray-tracing sources
    scn = oyens_scene_factory()
    scn.state.add_emitter()
    scn.state.ctx.clear_sources()
    with pytest.raises(ValueError, match="Ray-tracing engine has no sources!"):
        syn.validate_scene(scn)

    # Do the same for the capsules
    class TempMic:
        @property
        def n_capsules(self):
            return 5

    scn = oyens_scene_factory()
    scn.add_event(event_type="static")
    scn.state.microphones["asdf"] = TempMic()
    with pytest.raises(ValueError, match="Mismatching number of microphones"):
        syn.validate_scene(scn)


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


@pytest.mark.parametrize("duration", [20, 30])
def test_generate_dcase_2024_metadata(duration: int):
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
    dcase_out = syn.generate_dcase2024_metadata(scene)
    # Scene only has one listener, so we should only have one dataframe
    assert len(dcase_out) == 1
    dcase = dcase_out["mic000"]
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
    # Altering one of the class indices: should lead to an error as we expect this to be an int
    scene.events["event000"].class_id = "asdf"
    with pytest.raises(ValueError):
        _ = syn.generate_dcase2024_metadata(scene)
