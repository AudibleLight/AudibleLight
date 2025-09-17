#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/augmentation.py"""

import librosa
import numpy as np
import pytest
from scipy import stats

from audiblelight import config, custom_types, utils
from audiblelight.augmentation import (
    ALL_EVENT_AUGMENTATIONS,
    Augmentation,
    Chorus,
    Clipping,
    Compressor,
    Deemphasis,
    Delay,
    Distortion,
    EventAugmentation,
    Fade,
    Gain,
    GSMFullRateCompressor,
    HighpassFilter,
    LibriSpeechBasic,
    Limiter,
    LowpassFilter,
    MP3Compressor,
    MultibandEqualizer,
    Phaser,
    SpecAugmentPolicy,
    SpeedUp,
    TimeFrequencyMasking,
    TimeWarp,
    TimeWarpDuplicate,
    TimeWarpRemove,
    TimeWarpReverse,
    TimeWarpSilence,
    validate_event_augmentation,
)
from audiblelight.event import Event
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

        if isinstance(param_val, custom_types.Numeric):
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
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=config.SAMPLE_RATE)

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

    # testing equality
    assert reloaded == fx_init
    assert reloaded != 123
    assert reloaded != "another dtype"

    # Should be different to other audiblelight types, including the same augmentation with a different sample rate
    assert reloaded != Event(alias="123", filepath=utils_tests.TEST_AUDIOS[0])
    assert reloaded != fx_class(sample_rate=8000)


def test_load_from_dict_bad():
    cls = Augmentation(sample_rate=8000)

    outp = cls.to_dict()
    outp.pop("name")
    with pytest.raises(
        KeyError, match="Augmentation name must be specified in dictionary"
    ):
        _ = Augmentation.from_dict(outp)

    outp["name"] = "breaker"
    with pytest.raises(KeyError, match="Augmentation class breaker not found"):
        _ = Augmentation.from_dict(outp)


# Tests all combinations of fade shapes
@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_MUSICS[:3])
@pytest.mark.parametrize("fade_in_shape", Fade.FADE_SHAPES + [None])
@pytest.mark.parametrize("fade_out_shape", Fade.FADE_SHAPES + [None])
def test_fade(fade_in_shape, fade_out_shape, audio_fpath):
    # Load up the audio file in librosa
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=config.SAMPLE_RATE)

    # Process the audio with the augmentation
    #  Length of the fade in and fade out should always be 5 seconds
    fader = Fade(
        fade_in_len=5,
        fade_out_len=5,
        fade_in_shape=fade_in_shape,
        fade_out_shape=fade_out_shape,
    )
    out = fader(loaded)

    # Consider final sample and average volume of final N seconds
    if fader.fade_out_shape != "none":
        assert np.isclose(out[-1], 0.0, atol=utils.SMALL)
        fade_time = round(config.SAMPLE_RATE * fader.fade_out_len)
        assert np.mean(np.abs(out[-fade_time:])) <= np.mean(np.abs(loaded[-fade_time:]))

    # Consider first sample and average volume of first N seconds
    if fader.fade_in_shape != "none":
        assert np.isclose(out[0], 0.0, atol=utils.SMALL)
        fade_time = round(config.SAMPLE_RATE * fader.fade_in_len)
        assert np.mean(np.abs(out[:fade_time])) <= np.mean(np.abs(loaded[:fade_time]))


@pytest.mark.parametrize(
    "params",
    [
        dict(fade_in_shape="bad"),
        dict(fade_out_shape="bad"),
        dict(fade_in_shape="bad", fade_out_shape="bad"),
    ],
)
def test_bad_fade_shape(params):
    with pytest.raises(ValueError):
        _ = Fade(**params)


@pytest.mark.parametrize(
    "sample_rate,raises", [(44100, False), (8001, True), (11025, False), (48000, False)]
)
def test_mp3_compressor(sample_rate, raises):
    if raises:
        with pytest.raises(ValueError):
            _ = MP3Compressor(sample_rate)
    else:
        aug = MP3Compressor(sample_rate)
        assert aug.sample_rate == sample_rate


@pytest.mark.parametrize("threshold_db", [100, -100, -10])
@pytest.mark.parametrize("augmentation_class", [Limiter, Clipping, Compressor])
def test_threshold_db(threshold_db, augmentation_class):
    # Threshold dB value should always be negative
    cls = augmentation_class(threshold_db=threshold_db)
    assert cls.threshold_db == -abs(threshold_db)


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_MUSICS[:3])
@pytest.mark.parametrize("stretch_factor", [0.5, 1.0, 1.5])
def test_speed_up(audio_fpath, stretch_factor):
    # Load up the audio file in librosa
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=config.SAMPLE_RATE)

    # Process the audio
    cls = SpeedUp(stretch_factor=stretch_factor)
    out = cls(loaded)

    # Should be the same shape
    try:
        utils.validate_shape(loaded.shape, out.shape)
    except ValueError as e:
        pytest.fail(e)

    # With a stretch factor of 1.0, audio should be equal
    if stretch_factor == 1.0:
        assert np.array_equal(out, loaded)

    # With a stretch factor of more than 1.0, audio should be right padded with zeros
    elif stretch_factor > 1.0:
        assert np.array_equal(out[-100:], np.zeros(100))
        assert not np.array_equal(loaded[-100:], np.zeros(100))
        assert not np.array_equal(out, loaded)

    else:
        assert not np.array_equal(out, loaded)


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_MUSICS[:3])
@pytest.mark.parametrize(
    "augmentation_class",
    [TimeWarpSilence, TimeWarpReverse, TimeWarpRemove, TimeWarpDuplicate, TimeWarp],
)
@pytest.mark.parametrize(
    "params",
    [dict(fps=0.01, prob=0.5), dict(fps=10, prob=0.0), dict(prob=0.5, fps=2.5)],
)
def test_timewarp_effects(audio_fpath, augmentation_class, params):
    # Load up the audio file in librosa
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=config.SAMPLE_RATE)

    # Process the audio
    cls = augmentation_class(**params)
    out = cls(loaded)

    # Should be the same shape output
    try:
        utils.validate_shape(loaded.shape, out.shape)
    except ValueError as e:
        pytest.fail(e)


@pytest.mark.parametrize(
    "augmentation_class",
    [TimeWarpSilence, TimeWarpReverse, TimeWarpRemove, TimeWarpDuplicate, TimeWarp],
)
def test_timewarp_bad(augmentation_class):
    with pytest.raises(ValueError, match="Expected fps to be greater than 0"):
        _ = augmentation_class(fps=0, sample_rate=config.SAMPLE_RATE)


def test_magic_methods():
    cls = Augmentation(sample_rate=config.SAMPLE_RATE)

    # test magic methods
    assert isinstance(cls.__repr__(), str)
    assert isinstance(cls.__str__(), str)
    assert len(cls) == 1
    assert len(list(iter(cls))) == 1


@pytest.mark.parametrize(
    "params,raises",
    [
        (dict(), False),
        (
            dict(
                n_bands=5,
                gain_db=[1, 2, 3, 4, 5],
                q=lambda: np.random.rand(),
                cutoff_frequency_hz=500,
            ),
            False,
        ),
        (dict(n_bands=3, gain_db=[1, 2]), True),
        (dict(n_bands=3, gain_db="123"), True),
    ],
)
def test_multiband_equalizer(params, raises):
    if raises:
        with pytest.raises((TypeError, ValueError)):
            _ = MultibandEqualizer(**params)

    else:
        cls = MultibandEqualizer(**params)
        audio = np.random.rand(1000)
        out = cls(audio)
        assert not np.array_equal(out, audio)


@pytest.mark.parametrize(
    "override,raises",
    [(None, False), (123, False), (lambda: np.random.rand(), False), ("asdf", True)],
)
def test_sample_value(override, raises):
    params = dict(override=override, default_dist=lambda: np.random.rand())

    if raises:
        with pytest.raises(TypeError):
            _ = Augmentation().sample_value(**params)
    else:
        out = Augmentation().sample_value(**params)
        assert isinstance(out, custom_types.Numeric)


def test_validate_event_augmentation():
    # Not callable
    with pytest.raises(ValueError, match="Augmentation object must be callable"):
        validate_event_augmentation(123)

    # A type, not an instance
    with pytest.raises(
        ValueError, match="Augmentation object must be an instance of a class"
    ):
        validate_event_augmentation(Distortion)

    # Not a subclass of EventAugmentation
    with pytest.raises(ValueError, match="Augmentation object must be a subclass"):
        validate_event_augmentation(Augmentation())

    # Does not have attributes
    temp = Distortion()
    del temp.fx
    with pytest.raises(
        AttributeError, match="Augmentation object must have 'fx' attribute"
    ):
        validate_event_augmentation(temp)

    # Different augmentation type
    temp.fx = lambda x: x
    temp.AUGMENTATION_TYPE = "bad"
    with pytest.raises(ValueError, match="Augmentation type must be 'event'"):
        validate_event_augmentation(temp)


@pytest.mark.parametrize("policy", ["LB", "LD", "SM", "SS"])
def test_time_frequency_masking(policy):
    masker = TimeFrequencyMasking(policy=policy, sample_rate=22050)

    # Generate some dummy audio: function should natively handle mono audio
    input_audio = np.random.rand(44100)

    # Process with the policy: this returns a WAVEFORM
    outp = masker(input_audio)

    # Check output: should have same dims to input, must be finite, but shouldn't be the same audio!
    assert isinstance(outp, np.ndarray)
    assert outp.shape == input_audio.shape
    assert np.isfinite(outp).all()
    assert not np.array_equal(outp, input_audio)

    # Check to_dict functionality
    as_dict = masker.to_dict()
    assert Augmentation.from_dict(as_dict) == masker


@pytest.mark.parametrize("audio_fpath", utils_tests.TEST_AUDIOS[:5])
def test_time_frequency_masking_spectrogram(audio_fpath: str):
    # Load up the audio file in librosa
    loaded, _ = librosa.load(audio_fpath, mono=True, sr=22050, duration=10)

    # Tile the audio to four channel
    input_audio = np.array([loaded, loaded, loaded, loaded])

    # Use an aggressive policy with lots of bands
    masker = TimeFrequencyMasking(
        policy={
            "F": 27,
            "m_F": 100,
            "T": 70,
            "m_T": 100,
        },
        sample_rate=22050,
        return_spectrogram=True,
        replace_with_zero=True,
        mel_kwargs=dict(hop_length=64, win_length=128),
    )

    # Process with the policy: this returns a SPECTROGRAM
    outp = masker(input_audio)
    assert outp.shape[0] == 4  # expect four channels

    # Iterate over the four channel audio
    for channel in outp:
        # Iterate over all the "columns": at least one should be all zeroes (because of the time masking)
        assert np.any(np.all(channel == 0, axis=0))
        # Do the same for rows (frequency masking)
        assert np.any(np.all(channel == 0, axis=1))


@pytest.mark.parametrize(
    "policy,raises",
    [
        ("LB", False),
        (["LB", "LD"], False),
        (["LB", LibriSpeechBasic()], False),
        (LibriSpeechBasic, False),
        ({"F": 1, "m_F": 3, "T": 100, "m_T": 100}, False),
        (
            [
                {"F": 1, "m_F": 3, "T": 100, "m_T": 100},
                "LB",
                LibriSpeechBasic,
                LibriSpeechBasic(),
            ],
            False,
        ),
        (None, False),
        ("ASDASD", KeyError),
        (123, TypeError),
        ([], ValueError),
    ],
)
def test_time_frequency_masking_policy(policy, raises):
    if not raises:
        masker = TimeFrequencyMasking(policy=policy, sample_rate=22050)
        assert issubclass(type(masker.policy), SpecAugmentPolicy)
    else:
        with pytest.raises(raises):
            _ = TimeFrequencyMasking(policy=policy, sample_rate=22050)
