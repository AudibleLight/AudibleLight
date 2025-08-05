#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/utils.py"""

import os
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import pytest
from scipy import stats

from audiblelight import utils

EXISTING_FILE = utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb"


@pytest.mark.parametrize(
    "spherical, expected",
    [
        # azimuth=0°, polar=0°, r=1 -> (0, 0, 1)
        (np.array([[0, 0, 1]]), np.array([[0, 0, 1]])),
        # azimuth=0°, polar=90°, r=1 -> (1, 0, 0)
        (np.array([[0, 90, 1]]), np.array([[1, 0, 0]])),
        # azimuth=90°, polar=90°, r=1 -> (0, 1, 0)
        (np.array([[90, 90, 1]]), np.array([[0, 1, 0]])),
        # azimuth=180°, polar=90°, r=1 -> (-1, 0, 0)
        (np.array([[180, 90, 1]]), np.array([[-1, 0, 0]])),
        # azimuth=0°, polar=180°, r=1 -> (0, 0, -1)
        (np.array([[0, 180, 1]]), np.array([[0, 0, -1]])),
        # Multiple points
        (
            np.array(
                [
                    [0, 0, 1],  # +Z
                    [0, 90, 1],  # +X
                    [90, 90, 1],  # +Y
                    [180, 90, 1],  # -X
                    [0, 180, 1],  # -Z
                ]
            ),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
        ),
    ],
)
def test_polar_to_cartesian(spherical, expected):
    result = utils.polar_to_cartesian(spherical)
    assert np.allclose(result, expected, atol=1e-4)
    # Go the other way
    assert np.allclose(utils.cartesian_to_polar(result), spherical, atol=1e-4)


@pytest.mark.parametrize(
    "cartesian, expected",
    [
        # (x, y, z) -> (azimuth°, polar°, r)
        (np.array([[0, 0, 1]]), np.array([[0, 0, 1]])),  # +Z axis
        (np.array([[1, 0, 0]]), np.array([[0, 90, 1]])),  # +X axis
        (np.array([[0, 1, 0]]), np.array([[90, 90, 1]])),  # +Y axis
        (np.array([[-1, 0, 0]]), np.array([[180, 90, 1]])),  # -X axis
        (np.array([[0, 0, -1]]), np.array([[0, 180, 1]])),  # -Z axis
        (
            np.array(
                [
                    [0, 0, 1],  # +Z
                    [1, 0, 0],  # +X
                    [0, 1, 0],  # +Y
                    [-1, 0, 0],  # -X
                    [0, 0, -1],  # -Z
                ]
            ),
            np.array(
                [
                    [0, 0, 1],
                    [0, 90, 1],
                    [90, 90, 1],
                    [180, 90, 1],
                    [0, 180, 1],
                ]
            ),
        ),
    ],
)
def test_cartesian_to_polar(cartesian, expected):
    result = utils.cartesian_to_polar(cartesian)
    np.testing.assert_allclose(result, expected, atol=1e-4)
    # Go the other way
    assert np.allclose(utils.polar_to_cartesian(result), cartesian, atol=1e-4)


@pytest.mark.parametrize(
    "fpath,expected",
    [
        (EXISTING_FILE, Path(EXISTING_FILE)),
        (Path(EXISTING_FILE), Path(EXISTING_FILE)),
        ("a/broken/filepath", FileNotFoundError),
        (123456, TypeError),
    ],
)
def test_sanitise_filepath(fpath, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = utils.sanitise_filepath(fpath)
    else:
        assert expected == utils.sanitise_filepath(fpath)


@pytest.mark.parametrize(
    "num,expected", [(1, 1.0), (0.5, 0.5), (-1 / 3, ValueError), ("asdf", TypeError)]
)
def test_sanitise_positive_number(num, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = utils.sanitise_positive_number(num)
    else:
        assert expected == utils.sanitise_positive_number(num)


@pytest.mark.parametrize(
    "coords,expected",
    [
        ([0.5, 0.5, 0.5], np.array([0.5, 0.5, 0.5])),
        (np.array([0.1, 0.1]), ValueError),
        ("asdf", TypeError),
    ],
)
def test_sanitise_coordinates(coords, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = utils.sanitise_coordinates(coords)
    else:
        assert np.array_equal(expected, utils.sanitise_coordinates(coords))


@pytest.mark.parametrize(
    "dist,error",
    [
        # Scipy distributions should work ok out-of-the-box
        (stats.norm(0.5, 1.5), None),
        (stats.truncnorm(0.5, 1.5, 0.2, 1.7), None),
        (stats.uniform(0.0, 1.0), None),
        # Callable that returns a random value -- ok
        (lambda: np.random.uniform(0.0, 1.0), None),
        # Callable that will error out
        (lambda: 1 / 0, TypeError),
        # Callable that returns a string, or just a string -- nope
        (lambda: "asdf", TypeError),
        ("asdf", TypeError),
    ],
)
def test_sanitise_distribution(dist, error):
    if not error:
        # Sanitised distributions should have a `rvs` method which, when called, returns a number
        #  This is identical to how SciPy handles distributions
        #  We wrap non-SciPy objects in a `DistributionWrapper` class that adds the `rvs` method
        sanitised = utils.sanitise_distribution(dist)
        assert hasattr(sanitised, "rvs")
        assert callable(sanitised.rvs)
        assert isinstance(sanitised.rvs(), (int, float, complex))
    else:
        with pytest.raises(error):
            _ = utils.sanitise_distribution(dist)


@pytest.mark.parametrize(
    "prefix,objects,expected",
    [("tmp", {"tmp000": 123, "tmp001": 321}, "tmp002"), ("empty", {}, "empty000")],
)
def test_get_default_alias_(prefix, objects, expected):
    actual = utils.get_default_alias(prefix, objects)
    assert actual == expected


def _tmp(*_):
    return


@pytest.mark.parametrize(
    "func, kwargs, should_raise, expected_exception",
    [
        (lambda x=1, y=2: x + y, {"x": 5}, False, None),  # Valid kwarg
        (lambda x=1, y=2: x + y, {"z": 5}, True, AttributeError),  # Invalid kwarg
        (
            lambda **kwargs: sum(kwargs.values()),
            {"a": 1},
            False,
            None,
        ),  # Accepts arbitrary kwargs
        ("not_a_function", {"x": 1}, True, TypeError),  # Not a callable
        (lambda x, y: x + y, {}, False, None),  # No kwargs but valid empty call
        (lambda *, a=1: a, {"a": 2}, False, None),  # Keyword-only argument
        (lambda a, b: a + b, {"c": 3}, True, AttributeError),  # Invalid kwarg
        (_tmp, {"a": 1}, True, ValueError),  # does not take kwargs
    ],
)
def test_validate_kwargs(func, kwargs, should_raise, expected_exception):
    if should_raise:
        with pytest.raises(expected_exception):
            utils.validate_kwargs(func, **kwargs)
    else:
        utils.validate_kwargs(func, **kwargs)


@pytest.mark.parametrize(
    "directory,raises",
    [
        (
            utils.get_project_root() / "tests",
            False,
        ),
        (
            "asdfasgsfhdsh",
            FileNotFoundError,
        ),
        (
            utils.get_project_root() / "tests/test_utils.py",
            ValueError,
        ),
    ],
)
def test_list_all_directories(directory, raises):
    if raises:
        with pytest.raises(raises):
            _ = utils.list_all_directories(directory)
    else:
        out = utils.list_all_directories(directory)
        assert len(out) >= 1
        for path in out:
            assert os.path.isdir(path)
            assert isinstance(path, str)


@pytest.mark.parametrize(
    "directory,expected",
    [(utils.get_project_root() / "tests", ["femaleSpeech", "music", "meshes"])],
)
def test_list_deepest_directories(directory, expected):
    out = utils.list_deepest_directories(directory)
    assert isinstance(out, list)
    for expect in expected:
        assert any([expect in actual for actual in out])


@pytest.mark.parametrize(
    "directory,expected",
    [(utils.get_project_root() / "tests", ["femaleSpeech", "music", "meshes"])],
)
def test_list_innermost_directory_names(directory, expected):
    out = utils.list_innermost_directory_names(directory)
    assert isinstance(out, list)
    for expect in expected:
        assert any([expect in actual for actual in out])


@pytest.mark.parametrize(
    "directory,expected",
    [(utils.get_project_root() / "tests", ["femaleSpeech", "music", "meshes"])],
)
def test_list_innermost_directory_names_unique(directory, expected):
    out = utils.list_innermost_directory_names_unique(directory)
    assert isinstance(out, set)
    for expect in expected:
        assert any([expect in actual for actual in out])


@pytest.mark.parametrize(
    "shape_a,shape_b,raises",
    [
        # Matching shapes
        ((3, 4), (3, 4), False),
        ((5,), (5,), False),
        ((3, 4, 5), (3, 4, 5), False),
        # Non-matching shapes
        ((3, 4), (3, 5), True),
        ((3,), (4,), True),
        ((3, 4, 5), (3, 4, 6), True),
        ((1,), (2, 1), True),
    ],
)
def test_validate_shape(
    shape_a: tuple[Union[int, None]], shape_b: tuple[Union[int, None]], raises: bool
):
    if raises:
        with pytest.raises(ValueError):
            utils.validate_shape(shape_a, shape_b)
    else:
        utils.validate_shape(shape_a, shape_b)


@pytest.mark.parametrize(
    "distribution,override,raises",
    [
        (lambda: 1, None, False),
        (lambda: 1, 2, False),
        (None, None, ValueError),
        (None, "asdf", TypeError),
    ],
)
def test_sample_distribution(distribution, override, raises):
    if not raises:
        out = utils.sample_distribution(distribution, override)
        assert isinstance(out, utils.Numeric)
    else:
        with pytest.raises(raises):
            _ = utils.sample_distribution(distribution, override)


@pytest.mark.parametrize("seed", [utils.SEED, 111, 123, 156])
def test_seed_everything(seed):
    utils.seed_everything(seed)
    in1 = np.random.rand(10)
    utils.seed_everything(seed)
    in2 = np.random.rand(10)
    assert np.array_equal(in1, in2)


@pytest.mark.parametrize(
    "iterables,expected",
    [
        [[[1, 2, 3], [3, 2, 1], [10, 10, 10]], True],
        [["asdf", "fdsa", [1, 2, 3, 4]], True],
        [["asdf", "ds"], False],
        [[{1, 2, 3}, "asd", [1, 2, 3]], True],
        [["asdf", [3, 3], range(5)], False],
    ],
)
def test_check_all_lens_equal(iterables, expected):
    assert utils.check_all_lens_equal(*iterables) == expected


class Tmp:
    @staticmethod
    def to_dict():
        return {"asdf": 1}


@pytest.mark.parametrize(
    "cls, raises",
    [
        (Tmp, False),
        (str, True),
        (int, True),
    ],
)
def test_repr_as_json(cls, raises):
    if raises:
        with pytest.raises(AttributeError):
            _ = utils.repr_as_json(cls)
    else:
        out = utils.repr_as_json(cls)
        assert isinstance(out, str)


@pytest.mark.parametrize(
    "prefix, objects, expected, raises",
    [
        ("test", {"test000": 123}, "test001", False),
        ("test", {"test000": 123, "test001": 321}, "test002", False),
        ("test", {"test001": 123}, None, True),
    ],
)
def test_get_default_alias(prefix, objects, expected, raises):
    if raises:
        with pytest.raises(KeyError):
            _ = utils.get_default_alias(prefix, objects)
    else:
        out = utils.get_default_alias(prefix, objects)
        assert out == expected


@pytest.mark.parametrize(
    "audio_input, expect_warning, expect_normalized",
    [
        (np.array([0.0, 0.5, -0.5]), False, False),  # Normal audio, within range
        (np.array([1.0, -1.0, 0.999]), False, False),  # Edge values, no normalization
    ],
)
def test_write_wav(audio_input, expect_warning, expect_normalized, caplog):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test.wav"

        # Capture logging output
        with caplog.at_level("WARNING"):
            utils.write_wav(audio_input, str(outpath))

        # Check if output file was created
        assert outpath.exists()

        # Check warning presence
        if expect_warning:
            assert any("warning" in rec.levelname.lower() for rec in caplog.records)
        else:
            assert all("warning" not in rec.levelname.lower() for rec in caplog.records)

        # Read back WAV and check normalization
        with wave.open(str(outpath), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)

        if expect_normalized:
            assert np.max(np.abs(data)) == 32767
