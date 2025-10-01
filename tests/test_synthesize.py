#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/synthesize.py"""


from time import time

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


@pytest.mark.parametrize("n_emitters", list(range(1, 4)))
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

    # Check audio
    final_audio = ev.spatial_audio["mic000"]
    assert isinstance(final_audio, np.ndarray)
    assert final_audio.shape[0] == 4
    assert final_audio.ndim == 2

    # Audio should be normalized to match target dB level
    # Check that audio dB ~= (ref_db + event_snr)
    r_output_scaled_actual = syn.estimate_signal_rms(final_audio)
    r_output_scaled_db = 20 * np.log10(r_output_scaled_actual)
    error = abs(r_output_scaled_db - (ev.snr + oyens_scene_no_overlap.ref_db))
    assert pytest.approx(error, abs=3) == 0.0

    # Audio should not clip
    assert np.max(np.abs(final_audio)) <= 1.0


@pytest.mark.parametrize(
    "n_events",
    [
        1,
        2,
    ],
)
def test_render_scene_audio_from_static_events(n_events: int, oyens_scene_no_overlap):
    oyens_scene_no_overlap.clear_events()
    # Add static sources in with a given SNR
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(event_type="static")

    syn.validate_scene(oyens_scene_no_overlap)
    init_time = time()
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap, ignore_cache=True)
    no_cache_time = time() - init_time
    assert len(oyens_scene_no_overlap.events) == n_events

    # Grab audio for every event
    for event_alias, event in oyens_scene_no_overlap.events.items():
        assert isinstance(event.spatial_audio["mic000"], np.ndarray)
        final_audio = event.spatial_audio["mic000"]

        # Number of channels should be same as microphone, number of samples should be same as audio
        n_channels, n_samples = final_audio.shape
        assert n_channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
        assert n_samples == event.audio.shape[-1]

        # Audio should be normalized to match target dB level
        # Check that audio dB ~= (ref_db + event_snr)
        target_db = event.snr + oyens_scene_no_overlap.ref_db
        r_output_scaled_actual = syn.estimate_signal_rms(final_audio)
        r_output_scaled_db = 20 * np.log10(r_output_scaled_actual)
        assert pytest.approx(r_output_scaled_db, abs=3) == target_db

        # Audio should not clip
        assert np.max(np.abs(final_audio)) <= 1.0

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

    # Grab audio for every event
    for event_alias, event in oyens_scene_no_overlap.events.items():
        assert isinstance(event.spatial_audio["mic000"], np.ndarray)
        final_audio = event.spatial_audio["mic000"]

        # Number of channels should be same as microphone, number of samples should be same as audio
        n_channels, n_samples = final_audio.shape
        assert n_channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
        assert n_samples == event.audio.shape[-1]

        # Audio should be normalized to match target dB level
        # Check that audio dB ~= (ref_db + event_snr)
        r_output_scaled_actual = syn.estimate_signal_rms(final_audio)
        r_output_scaled_db = 20 * np.log10(r_output_scaled_actual)
        error = abs(r_output_scaled_db - (event.snr + oyens_scene_no_overlap.ref_db))
        assert pytest.approx(error, abs=3) == 0.0

        # Audio should not clip
        assert np.max(np.abs(final_audio)) <= 1.0


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
    oyens_scene_no_overlap.allow_duplicate_audios = True

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
        def n_listeners(self):
            return 5

    scn = oyens_scene_factory()
    scn.add_event(event_type="static")
    scn.state.microphones["asdf"] = TempMic()
    with pytest.raises(ValueError, match="Mismatching number of microphones"):
        syn.validate_scene(scn)


@pytest.mark.parametrize(
    "ambience_kws",
    [
        # Use Scene noise floor, ref_db is None
        dict(
            noise="gaussian",
        ),
        # Different noise floor to Scene
        dict(noise="pink", ref_db=-40),
        # Use audio file + different noise floor to scene
        dict(filepath=utils_tests.TEST_MUSICS[0], ref_db=-30),
    ],
)
def test_render_ambience(ambience_kws, oyens_scene_no_overlap):
    # Add ambience and a single event (so things don't break)
    oyens_scene_no_overlap.add_ambience(**ambience_kws, alias="tester")
    oyens_scene_no_overlap.add_event(
        event_type="static", duration=1.0, event_start=10.0
    )

    # Render the audio
    syn.render_audio_for_all_scene_events(oyens_scene_no_overlap)
    syn.generate_scene_audio_from_events(oyens_scene_no_overlap)

    # Check the audio is present
    amb = oyens_scene_no_overlap.get_ambience("tester")
    assert isinstance(amb.spatial_audio["mic000"], np.ndarray)
    final_audio = amb.spatial_audio["mic000"]

    # Number of channels should be same as microphone, number of samples should be same as audio
    n_channels, n_samples = final_audio.shape
    assert n_channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
    assert n_samples == amb.load_ambience().shape[-1]

    # Audio should be normalized to match target dB level
    # Check that mean(abs(audio)) in dB == target_db
    r_output_scaled_actual = syn.estimate_signal_rms(final_audio)
    r_output_scaled_db = 20 * np.log10(r_output_scaled_actual)
    # Default to using Scene noise floor if not provided explicitly
    target_db = ambience_kws.get("ref_db", oyens_scene_no_overlap.ref_db)
    error = abs(r_output_scaled_db - target_db)
    assert pytest.approx(error, abs=3) == 0.0

    # Audio should not clip
    assert np.max(np.abs(final_audio)) <= 1.0
