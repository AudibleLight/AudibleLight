#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/synthesize.py"""


import numpy as np
import pytest

import audiblelight.synthesize as syn
from audiblelight import utils


@pytest.mark.parametrize(
    "audio, ir, expected_shape",
    [
        (np.ones(4), np.ones((4, 1)), (4, 1)),    # Mono audio, single-channel IR, both equal length
        (np.ones(5), np.ones((5, 4)), (5, 4)),     # Mono audio, multi-channel IR
        (np.ones(6), np.ones((3, 1)), (6, 1)),    # Shorter IR, triggers padding
        (np.ones(4), np.ones((8, 2)), (4, 2)),    # Longer IR (result gets truncated)
        # replicate FOA audio
        (
            np.random.rand(utils.SAMPLE_RATE * 10),
            np.random.rand(utils.SAMPLE_RATE * 2, 4),
            (utils.SAMPLE_RATE * 10, 4)
        ),
    ]
)
def test_time_invariant_convolution_valid(audio, ir, expected_shape):
    output = syn.time_invariant_convolution(audio, ir)
    assert isinstance(output, np.ndarray)
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "audio, ir, error_message",
    [
        (np.ones((5, 2)), np.ones((5, 4)), "Only mono input is supported"),    # audio not mono
        (np.ones(5), np.ones(5), "Expected shape of IR should be"),    # IR not 2D
        (np.ones(5), np.ones((5, 1, 1)), "Expected shape of IR should be"),    # IR not 2D
    ]
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
    ]
)
def test_apply_snr(x, snr, expected_max):
    result = syn.apply_snr(x, snr)
    actual_max = np.max(np.abs(result))
    assert np.isclose(actual_max, expected_max, rtol=1e-5), f"Expected max {expected_max}, got {actual_max}"
