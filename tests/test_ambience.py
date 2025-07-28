#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/ambience.py, many of which are adapted from `colorednoise`"""

import numpy as np
import pytest

from audiblelight.ambience import powerlaw_psd_gaussian, _parse_beta, Ambience
from audiblelight import utils


@pytest.mark.parametrize("shape", [2, 16, 500, 1000])
def test_powerlaw_psd_gaussian_scalar_output_shape(shape):
    n = powerlaw_psd_gaussian(1, shape)
    assert n.shape == (shape,)


@pytest.mark.parametrize("shape", [(1, 1000), (6, 600), (4, 44100), (8, 100000)])
def test_powerlaw_psd_gaussian_vector_output_shape(shape):
    n = powerlaw_psd_gaussian(1, shape)
    assert n.shape == shape


def test_powerlaw_psd_gaussian_output_finite():
    n = powerlaw_psd_gaussian(1, 16, fmin=0.1)
    assert np.isfinite(n).all()


@pytest.mark.parametrize("exponent", [0.5, 1, 2])
def test_var_distribution(exponent):
    size = (100, 2 ** 16)
    fmin = 0
    y = powerlaw_psd_gaussian(exponent, size, fmin=fmin, seed=1)
    ystd = y.std(axis=-1)
    var_in = (abs(1 - ystd) < 3 * ystd.std()).mean()
    assert var_in > 0.95


@pytest.mark.parametrize("nsamples", [10, 11])
def test_small_sample_var(nsamples):
    ystd = powerlaw_psd_gaussian(0, (500, 500, nsamples), seed=1).std(axis=-1)
    assert (abs(1 - ystd) < 3 * ystd.std()).mean() > 0.95


@pytest.mark.parametrize("exponent", [0.5, 1, 2])
def test_slope_distribution(exponent):
    size = (100, 2 ** 16)
    fmin = 0
    y = powerlaw_psd_gaussian(exponent, size, fmin=fmin, seed=1)
    yfft = np.fft.fft(y)
    f = np.fft.fftfreq(y.shape[-1])
    m = f > 0
    fit, fcov = np.polyfit(np.log10(f[m]), np.log10(np.abs(yfft[..., m].T ** 2)),1, cov=True)
    slope_in = (exponent + fit[0] < 3 * np.sqrt(fcov[0, 0])).mean()
    assert slope_in > 0.95


def test_cumulative_scaling():
    n_repeats = 1000
    n_steps = 100
    y = powerlaw_psd_gaussian(0, (n_repeats, n_steps), seed=1)
    mean_squared_displacement = (y.sum(axis=-1) ** 2).mean(axis=0)
    standard_error = (y.sum(axis=-1) ** 2).std(axis=0) / np.sqrt(n_repeats)
    assert abs(n_steps - mean_squared_displacement) < 3 * standard_error


def test_random_state_reproducibility():
    exp = 1
    n = 5
    y1 = powerlaw_psd_gaussian(exp, n, seed=1)
    np.random.seed(123)  # Reset global RNG to make sure it doesn’t interfere
    y2 = powerlaw_psd_gaussian(exp, n, seed=1)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize(
    "color, expected",
    [
        ("white", 0),
        ("pink", 1),
        ("brown", 2),
        ("not-a-color", KeyError),
        (1.5, 1.5),
        (0, 0),
        (np.float64(2.0), 2.0),
        (set(), TypeError)
    ]
)
def test_parse_beta(color, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _parse_beta(color,)
    else:
        assert _parse_beta(color,) == expected


@pytest.mark.parametrize(
    "channels, duration, noise, filepath",
    [
        (4, 2, "white", None),
        (4, 2, 2.0, None),
        (2, 4, None, utils.get_project_root() / "tests/test_resources/soundevents/waterTap/95709.wav"),
        (1, 2, None, utils.get_project_root() / "tests/test_resources/soundevents/telephone/30085.wav"),
    ]
)
def test_ambience_cls(channels, duration, noise, filepath):
    cls = Ambience(channels, duration, noise=noise, filepath=filepath, alias="tester")
    assert isinstance(cls.to_dict(), dict)
    assert cls.load_ambience().shape == (channels, round(duration * cls.sample_rate))
