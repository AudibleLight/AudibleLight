#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for SOFA functionality inside audiblelight/worldstate.py"""

from unittest.mock import Mock

import numpy as np
import pytest

from audiblelight import utils
from audiblelight.worldstate import Emitter, WorldStateSOFA
from tests import utils_tests


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(position=None, keep_existing=True, alias="tmp"),
        dict(
            position=[2.5, 0.0, 0.0],
            keep_existing=False,
        ),
    ],
)
def test_add_emitter(kwargs, daga_space: WorldStateSOFA):
    # Add the emitter in with desired arguments
    daga_space.add_emitter(**kwargs)

    # Should be a dictionary with one emitter
    assert isinstance(daga_space.emitters, dict)
    assert daga_space.num_emitters == 1 == len(daga_space.emitters)

    # Get the desired emitter: should be the first element in the list
    emitter_alias = kwargs.get("alias", "src000")
    src = daga_space.get_emitter(kwargs.get("alias", "src000"), 0)

    # Should be an emitter object
    assert isinstance(src, Emitter)
    # Should have all the desired attributes
    assert src.alias == emitter_alias

    # Actual position should be within the SOFA file
    actual_pos = src.coordinates_absolute
    assert isinstance(actual_pos, np.ndarray)
    assert actual_pos in daga_space.get_source_positions()

    # If we've provided a position, actual one should be close
    expected_pos = kwargs.get("position", None)
    if expected_pos:
        assert np.allclose(expected_pos, src.coordinates_absolute, atol=utils.SMALL)

    # Emitter should have relative cartesian and polar coordinates
    assert isinstance(src.coordinates_relative_cartesian, dict)
    assert len(src.coordinates_relative_cartesian) == 1
    assert isinstance(src.coordinates_relative_polar, dict)
    assert len(src.coordinates_relative_cartesian) == 1


@pytest.mark.parametrize(
    "candidate_position,expected_idxs",
    [
        (np.array([0.6, 0.6, 0.6]), np.array([0])),
        (np.array([[0.6, 0.6, 0.6], [-0.6, -0.4, -0.3]]), np.array([0, 3])),
        (
            np.array([[1.2, 1.3, 1.3], [1.2, 1.3, 1.3], [-0.1, -0.1, -0.6]]),
            np.array([1, 1, 4]),
        ),
    ],
)
def test_get_nearest_source_idx(candidate_position, expected_idxs):
    ws = WorldStateSOFA(sofa=utils_tests.TEST_RESOURCES / "daga_foa.sofa")

    # Define some arbitrary source positions
    source_positions = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 1.5],
            [-0.5, -0.4, -0.3],
            [0.0, 0.0, -0.5],
        ]
    )

    # Hijack the method so that we return the desired source positions
    #  **not** the ones that are actually defined in the .sofa file
    ws.get_source_positions = Mock(return_value=source_positions)
    actual_idxs = ws.get_nearest_source_idx(candidate_position)

    # Should return the expected idxs
    assert np.array_equal(actual_idxs, expected_idxs)


@pytest.mark.parametrize("n_emitters", range(1, 4))
def test_simulate(n_emitters: int):
    # Create the WorldState
    ws = WorldStateSOFA(
        sofa=utils_tests.TEST_RESOURCES / "daga_foa.sofa",
        mic_alias="tester",
        sample_rate=22050,
    )
    ws.clear_emitters()

    # Add some emitters in
    for _ in range(n_emitters):
        ws.add_emitter(keep_existing=True)

    # Do the simulation and grab the IRs
    ws.simulate()
    irs = ws.irs["tester"]
    assert isinstance(irs, np.ndarray)

    # Expecting FOA for this SOFA file
    n_ch, n_emitters_expected, n_samples = irs.shape
    assert n_ch == 4
    assert n_emitters_expected == n_emitters
    assert n_samples > 1

    # IRs should be resampled to desired rate
    with ws.sofa() as sofa:
        orig_sr = int(sofa.getVariableValue("Data.SamplingRate"))
        orig_ir = np.array(sofa.getDataIR().data)
        orig_n_samples = orig_ir.shape[-1]
    expected_n_samples = round(orig_n_samples * ws.sample_rate / orig_sr)
    assert n_samples == expected_n_samples
