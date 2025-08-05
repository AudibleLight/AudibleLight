#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/synthesize.py"""


import numpy as np
import pytest

import audiblelight.synthesize as syn
from audiblelight import utils


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
        oyens_scene_no_overlap.add_event(
            event_type="static", emitter_kwargs=dict(keep_existing=True)
        )

    syn.validate_scene(oyens_scene_no_overlap)
    syn.render_scene_audio(oyens_scene_no_overlap)
    assert len(oyens_scene_no_overlap.events) == n_events

    for event_alias, event in oyens_scene_no_overlap.events.items():
        assert isinstance(event.spatial_audio, np.ndarray)
        n_channels, n_samples = event.spatial_audio.shape
        # Number of channels should be same as microphone, number of samples should be same as audio
        assert n_channels == oyens_scene_no_overlap.get_microphone("mic000").n_capsules
        assert n_samples == event.audio.shape[-1]


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
            filepath=utils.get_project_root()
            / "tests/test_resources/soundevents/music/000010.mp3",
            # Use predefined kwargs so rendering doesn't take ages
            event_type="moving",
            event_kwargs=dict(spatial_resolution=2, duration=1, spatial_velocity=1),
        )

    syn.validate_scene(oyens_scene_no_overlap)
    syn.render_scene_audio(oyens_scene_no_overlap)
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
    oyens_scene_no_overlap.clear_events()
    # Add both N static and N moving events
    for n_event in range(n_events):
        oyens_scene_no_overlap.add_event(
            event_type="static", emitter_kwargs=dict(keep_existing=True)
        )
        # Predefined kwargs so rendering doesn't take ages
        oyens_scene_no_overlap.add_event(
            event_type="moving",
            event_kwargs=dict(spatial_resolution=2, duration=1, spatial_velocity=1),
        )

    # Render the scene audio
    syn.validate_scene(oyens_scene_no_overlap)
    syn.render_scene_audio(oyens_scene_no_overlap)

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
