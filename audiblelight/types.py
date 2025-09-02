#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom data types, exceptions, objects, etc. used across the entire pipeline"""

from typing import Any, Callable, Protocol, Union

import numpy as np

# Numeric dtypes: useful for isinstance(x, ...) checking
NUMERIC_DTYPES = (
    int,
    float,
    complex,
    np.integer,
    np.floating,
)
# Used as a typehint
Numeric = Union[int, float, complex, np.integer, np.floating]
AUDIO_EXTS = ("wav", "mp3", "mpeg4", "m4a", "flac", "aac")


class DistributionLike(Protocol):
    """
    Typing protocol for any distribution-like object.

    Must expose an `rvs()` method that returns a single random variate as a float (or float-compatible number).
    """

    def rvs(self, *args: Any, **kwargs: Any) -> Numeric:
        pass


class DistributionWrapper:
    """
    Wraps a callable (e.g. a function) as a distribution-like object with an `rvs()` method.
    """

    def __init__(self, distribution: Callable):
        self.distribution = distribution

    def rvs(self, *_: Any, **__: Any) -> Numeric:
        return self.distribution()

    def __call__(self) -> Numeric:
        """Makes the wrapper itself callable like the original."""
        return self.rvs()
