#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/augmentation.py"""

import librosa
import numpy as np
import pytest
from scipy import stats

from audiblelight import utils
from audiblelight.augmentation import (
    ALL_EVENT_AUGMENTATIONS,
    Chorus,
    Compressor,
    Deemphasis,
    Delay,
    Distortion,
    EventAugmentation,
    Fade,
    Gain,
    GSMFullRateCompressor,
    HighpassFilter,
    LowpassFilter,
    MP3Compressor,
    MultibandEqualizer,
    Phaser,
    TimeWarpDuplicate,
)
from tests import utils_tests


@pytest.mark.parametrize(
    "fx_class,expecteds",
    [
        (
            LowpassFilter,
            dict(cutoff_frequency_hz=(LowpassFilter.MIN_FREQ, LowpassFilter.MAX_FREQ)),
        ),
        (
            HighpassFilter,
            dict(
                cutoff_frequency_hz=(HighpassFilter.MIN_FREQ, HighpassFilter.MAX_FREQ)
            ),
        ),
        (
            Compressor,
            dict(
                threshold_db=(Compressor.MIN_THRESHOLD_DB, Compressor.MAX_THRESHOLD_DB),
                attack_ms=(Compressor.MIN_ATTACK, Compressor.MAX_ATTACK),
                release_ms=(Compressor.MIN_RELEASE, Compressor.MAX_RELEASE),
                ratio=(min(Compressor.RATIOS), max(Compressor.RATIOS)),
            ),
        ),
        (
            Chorus,
            dict(
                rate_hz=(Chorus.MIN_RATE, Chorus.MAX_RATE),
                depth=(Chorus.MIN_DEPTH, Chorus.MAX_DEPTH),
                centre_delay_ms=(Chorus.MIN_DELAY, Chorus.MAX_DELAY),
                mix=(Chorus.MIN_MIX, Chorus.MAX_MIX),
                feedback=(Chorus.MIN_FEEDBACK, Chorus.MAX_FEEDBACK),
            ),
        ),
        (Distortion, dict(drive_db=(Distortion.MIN_DRIVE, Distortion.MAX_DRIVE))),
        (
            Phaser,
            dict(
                rate_hz=(Phaser.MIN_RATE, Phaser.MAX_RATE),
                depth=(Phaser.MIN_DEPTH, Phaser.MAX_DEPTH),
                centre_frequency_hz=(Phaser.MIN_FREQ, Phaser.MAX_FREQ),
                mix=(Phaser.MIN_MIX, Phaser.MAX_MIX),
                feedback=(Phaser.MIN_FEEDBACK, Phaser.MAX_FEEDBACK),
            ),
        ),
        (
            Delay,
            dict(
                delay_seconds=(Delay.MIN_DELAY, Delay.MAX_DELAY),
                mix=(Delay.MIN_MIX, Delay.MAX_MIX),
                feedback=(Delay.MIN_FEEDBACK, Delay.MAX_FEEDBACK),
            ),
        ),
        (Gain, dict(gain_db=(Gain.MIN_GAIN, Gain.MAX_GAIN))),
        (
            GSMFullRateCompressor,
            dict(
                quality=(
                    min(GSMFullRateCompressor.QUALITIES),
                    max(GSMFullRateCompressor.QUALITIES),
                )
            ),
        ),
        (
            MP3Compressor,
            dict(vbr_quality=(MP3Compressor.VBR_MIN, MP3Compressor.VBR_MAX)),
        ),
    ],
)
def test_parameter_defaults(fx_class, expecteds):
    """
    Test sampling FX parameters from default distributions
    """
    init_fx = fx_class()
    for param_name, (min_val, max_val) in expecteds.items():
        assert hasattr(init_fx, param_name)
        assert min_val <= getattr(init_fx, param_name) <= max_val


@pytest.mark.parametrize(
    "fx_class,params",
    [
        (LowpassFilter, dict(cutoff_frequency_hz=10000)),
        (HighpassFilter, dict(cutoff_frequency_hz=20)),
        (Compressor, dict(threshold_db=-50, attack_ms=0.01, release_ms=1.0, ratio=8)),
        (
            Chorus,
            dict(rate_hz=0.5, depth=0.1, centre_delay_ms=100, mix=0.1, feedback=0.6),
        ),
        (Distortion, dict(drive_db=20)),
        (
            Phaser,
            dict(
                rate_hz=0.5, depth=0.1, centre_frequency_hz=500, mix=0.5, feedback=0.4
            ),
        ),
        (Delay, dict(delay_seconds=0.5, mix=0.9, feedback=0.1)),
        (Gain, dict(gain_db=20)),
        (GSMFullRateCompressor, dict(quality=2)),
        (MP3Compressor, dict(vbr_quality=5.0)),
        (Deemphasis, dict(coef=0.5)),
        (TimeWarpDuplicate, dict(prob=0.9, fps=20)),
    ],
)
def test_parameter_provided(fx_class, params):
    """
    Test explicitly providing FX parameters
    """
    init_fx = fx_class(**params)
    for param_name, expected_val in params.items():
        assert hasattr(init_fx, param_name)
        assert getattr(init_fx, param_name) == expected_val


@pytest.mark.parametrize(
    "fx_class,params",
    [
        (LowpassFilter, dict(cutoff_frequency_hz=(1000, 2000))),
        (
            Chorus,
            dict(
                rate_hz=(0.5, 1.0),
                depth=(0.5, 0.6),
                centre_delay_ms=(100, 200),
                mix=(0.1, 0.9),
                feedback=(0.2, 0.3),
            ),
        ),
        (Distortion, dict(drive_db=(11, 15))),
    ],
)
def test_parameter_sampling(fx_class, params):
    """
    Test sampling from user-provided distributions
    """
    # Create distribution objects
    params_mod = {k: stats.uniform(a, b - a) for k, (a, b) in params.items()}
    init_fx = fx_class(**params_mod)
    for param_name, (min_val, max_val) in params.items():
        assert hasattr(init_fx, param_name)
        # Check within range of user-provided distribution
        assert min_val <= getattr(init_fx, param_name) <= max_val


@pytest.mark.parametrize(
    "params",
    [
        # Four bands, all other parameters randomly sampled
        dict(
            n_bands=4,
        ),
        # Four different bands, all with the same gain and Q
        dict(
            n_bands=4, gain_db=5.0, cutoff_frequency_hz=[1000, 2000, 3000, 4000], q=0.5
        ),
        # 1 band, all parameters defined
        dict(n_bands=1, gain_db=0.5, cutoff_frequency_hz=1000, q=0.25),
        # 3 bands, all parameters defined
        dict(
            n_bands=3,
            gain_db=np.array([0.5, 0.25, 1.5]),
            cutoff_frequency_hz=np.array([1000, 2000, 3000]),
            q=np.array([0.5, 1.5, 2.5]),
        ),
    ],
)
def test_equalizer(params):
    """
    Equalizer FX works a little differently as it is a list of pedalboard objects (PeakFilters)
    """
    init_fx = MultibandEqualizer(**params)

    # Should have set the number of bands correctly
    assert init_fx.n_bands == params["n_bands"]
    assert len(init_fx.fx) == params["n_bands"]

    # Iterate over all the other kwargs and check they're set correctly
    for param_name, param_val in params.items():
        if param_name == "n_bands":
            continue

        if isinstance(param_val, utils.Numeric):
            assert np.array_equal(
                getattr(init_fx, param_name),
                [param_val for _ in range(params["n_bands"])],
            )
        else:
            assert np.array_equal(getattr(init_fx, param_name), param_val)


@pytest.mark.parametrize("fx_class", ALL_EVENT_AUGMENTATIONS)
@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_AUDIOS[:5])
def test_process_audio(fx_class, audio_fpath):
    # Load up the audio file in librosa
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=utils.SAMPLE_RATE)

    # Initialise FX with default parameters and process the audio
    fx_init = fx_class()
    out = fx_init(loaded)

    # Should be an Event augmentation
    assert fx_class.AUGMENTATION_TYPE == "event"

    # Should be a child of EventAugmentation class
    assert issubclass(fx_class, EventAugmentation)

    # Should have required params
    assert hasattr(fx_init, "params")
    assert len(fx_init.params) > 0

    # Should be a numpy array with different values to initial
    #  Don't test when we have flaky FX,
    #  e.g. pitchshift can have a randomly sampled value of 0 semitones
    assert isinstance(out, np.ndarray)
    if not getattr(fx_init, "_FLAKY", False):
        assert not np.array_equal(out, loaded)

    # But should have the same shape
    try:
        utils.validate_shape(loaded.shape, out.shape)
    except ValueError as e:
        pytest.fail(e)


@pytest.mark.parametrize("fx_class", ALL_EVENT_AUGMENTATIONS)
def test_load_from_dict(fx_class):
    fx_init = fx_class()
    out_dict = fx_init.to_dict()
    reloaded = EventAugmentation.from_dict(out_dict)
    assert isinstance(reloaded, fx_class)
    assert reloaded == fx_init


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_MUSICS[:3])
@pytest.mark.parametrize(
    "fx_params",
    [
        # Fade-out only, long
        dict(fade_in_shape="none", fade_out_shape="linear", fade_out_len=5.0),
        # Fade-in only, long
        dict(fade_out_shape="none", fade_in_shape="linear", fade_in_len=5.0),
        # Both fades
        dict(
            fade_out_shape="linear",
            fade_in_shape="linear",
            fade_in_len=5.0,
            fade_out_len=5.0,
        ),
    ],
)
def test_fade(fx_params, audio_fpath):
    # Load up the audio file in librosa
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=utils.SAMPLE_RATE)

    # Process the audio with the augmentation
    fader = Fade(**fx_params)
    out = fader(loaded)

    # Consider final sample and average volume of final N seconds
    if fx_params["fade_out_shape"] != "none":
        assert out[-1] == 0.0
        fade_time = round(utils.SAMPLE_RATE * fx_params["fade_out_len"])
        assert np.mean(np.abs(out[-fade_time:])) < np.mean(np.abs(loaded[-fade_time:]))

    if fx_params["fade_in_shape"] != "none":
        assert out[0] == 0.0
        fade_time = round(utils.SAMPLE_RATE * fx_params["fade_in_len"])
        assert np.mean(np.abs(out[:fade_time])) < np.mean(np.abs(loaded[:fade_time]))
