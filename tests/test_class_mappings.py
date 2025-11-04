#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests mappings in audiblelight.class_mappings"""

import pytest

from audiblelight.class_mappings import (
    ALL_MAPPINGS,
    ClassMapping,
    DCASE2023Task3,
    sanitize_class_mapping,
)


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("dcase2023task3", DCASE2023Task3),
        (DCASE2023Task3, DCASE2023Task3),
        (DCASE2023Task3(), DCASE2023Task3),
        (dict(fake1=1, fake2=2), ClassMapping),
        (None, None),
        (1.0, TypeError),
        ("notinlist", ValueError),
    ],
)
def test_sanitize_mapping_input(inp: str, expected: object):
    if expected is None:
        actual = sanitize_class_mapping(inp)
        assert actual is None

    elif expected == TypeError:
        with pytest.raises(expected, match="Could not parse class mapping"):
            _ = sanitize_class_mapping(inp)

    elif expected == ValueError:
        with pytest.raises(expected, match="Cannot find class mapping"):
            _ = sanitize_class_mapping(inp)

    # should be instantiated
    else:
        actual = sanitize_class_mapping(inp)
        assert isinstance(actual, expected)
        assert issubclass(type(actual), ClassMapping)
        assert hasattr(actual, "mapping")
        assert hasattr(actual, "mapping_inverted")


@pytest.mark.parametrize(
    "filepath,expected_class,expected_idx,raises",
    [
        (
            "/AudibleLight/resources/soundevents/music/train/Pop/001649.mp3",
            "music",
            8,
            False,
        ),
        (
            "/AudibleLight/resources/soundevents/femaleSpeech/train/Female_speech_and_woman_speaking/109902.wav",
            "femaleSpeech",
            0,
            False,
        ),
        (
            "i/will/never/get/a/match/butitsok.wav",
            None,
            None,
            False,
        ),
        ("i/will/match/both/music/and/femaleSpeech/sowillfail.wav", None, None, True),
    ],
)
def test_infer_label_from_filepath(
    filepath, expected_class: str, expected_idx: int, raises: bool
):
    class_mapping = DCASE2023Task3()
    if raises:
        with pytest.raises(ValueError):
            _, __ = class_mapping.infer_label_idx_from_filepath(filepath)

    else:
        actual_idx, actual_cls = class_mapping.infer_label_idx_from_filepath(filepath)
        assert actual_cls == expected_class
        assert actual_idx == expected_idx


@pytest.mark.parametrize(
    "cls,idx", [("femaleSpeech", 0), ("footsteps", 6), ("music", 8)]
)
def test_infer_missing_values(cls, idx):
    mapper = DCASE2023Task3()

    # Try with label first
    actual_idx, _ = mapper.infer_missing_values(None, cls)
    assert actual_idx == idx

    # Then index second
    _, actual_cls = mapper.infer_missing_values(idx, None)
    assert actual_cls == cls


@pytest.mark.parametrize(
    "actual,expected", [("femaleSpeech", 0), (6, "footsteps"), ("asdf", KeyError)]
)
def test_getitem(actual, expected):
    if isinstance(expected, (str, int)):
        assert DCASE2023Task3()[actual] == expected
    else:
        with pytest.raises(KeyError):
            _ = DCASE2023Task3()[actual]


@pytest.mark.parametrize("mapper", ALL_MAPPINGS)
def test_magic_methods(mapper):
    mapper = mapper()
    assert isinstance(len(mapper), int)
    assert len(mapper) >= 0
    assert isinstance(mapper.to_dict(), dict)
    assert mapper.to_dict() == mapper.mapping


@pytest.mark.parametrize(
    "actual,expected,error",
    [
        ("asdf", TypeError, "Mapping must be a dict"),
        ({123: "asdf"}, TypeError, "Class name must be str"),
        ({"asdf": "asdf"}, TypeError, "Class index must be int"),
        ({"asdf": 0, "fdsa": 0}, ValueError, "Duplicate indices detected"),
        ({"asdf": 0, "fdsa": 3}, ValueError, "Indices must be contiguous"),
    ],
)
def test_validate_mapping(actual, expected, error):
    with pytest.raises(expected, match=error):
        _ = ClassMapping.from_dict(actual)
