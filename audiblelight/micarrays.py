#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements dataclasses for working with common microphone array types"""

from dataclasses import dataclass, field
from typing import Any, Type

import numpy as np
from loguru import logger

from audiblelight import utils

__all__ = [
    "sanitize_microphone_input",
    "MicArray",
    "Eigenmike32",
    "Eigenmike64",
    "MonoCapsule",
    "AmbeoVR",
    "MICARRAY_LIST"
]


@dataclass
class MicArray:
    """
    This is the base class for all microphone array types.

    Attributes:
        name (str): the name of the array.
        is_spherical (bool): whether the array is spherical. If False, positions_spherical will be None.

    Properties:
        coordinates_polar (np.array): the positions of the capsules on the array, given as azimuth, colatitude, radius
            (i.e., degrees, degrees, meters). Azimuth is measured counter-clockwise, where 0 == the front of the
            microphone (as given, e.g., by the manufacturer logo), colatitude/elevation increases from the top
            (0, away from the shaft) to the bottom (180, aligned with the shaft). When `is_spherical` is False, is None.
        coordinates_cartesian (np.array): the positions of the capsules in Cartesian (XYZ) coordinates, with distance
            measured using meters away from the center of the array
        coordinates_absolute (np.array): the absolute position of all capsules based on a provided center.
        n_capsules (int): number of capsules in the array
        capsule_names (list[str]): the names of the microphone capsules
    """

    name: str = ""
    is_spherical: bool = False
    irs: np.ndarray = field(default=None, init=False, repr=False)
    _coordinates_absolute: np.ndarray = field(default=None, init=False, repr=False)
    _coordinates_center: np.ndarray = field(default=None, init=False, repr=False)
    
    @property
    def coordinates_polar(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def coordinates_absolute(self) -> np.ndarray:
        if self._coordinates_absolute is None:
            raise NotImplementedError("Must call `.set_absolute_coordinates` first!")
        else:
            return (
                np.array(self._coordinates_absolute)
                if isinstance(self._coordinates_absolute, list)
                else self._coordinates_absolute
            )

    @property
    def coordinates_center(self) -> np.ndarray:
        if self._coordinates_center is None:
            raise NotImplementedError("Must call `.set_absolute_coordinates` first!")
        else:
            return (
                np.array(self._coordinates_center)
                if isinstance(self._coordinates_center, list)
                else self._coordinates_center
            )

    @property
    def n_capsules(self) -> int:
        return len(self.capsule_names)
    
    @property
    def capsule_names(self) -> list[str]:
        return []

    def set_absolute_coordinates(self, mic_center: np.ndarray) -> np.ndarray:
        """
        Calculates absolute position of all microphone capsules based on a provided center.

        The center should be in cartesian coordinates with the form (XYZ), with units in meters.
        """
        self._coordinates_center = mic_center
        self._coordinates_absolute = self.coordinates_cartesian + utils.coerce2d(self._coordinates_center)
        return self._coordinates_absolute

    def __len__(self) -> int:
        return self.n_capsules

    def __repr__(self) -> str:
        return utils.repr_as_json(self)

    def __str__(self) -> str:
        return f"Microphone array '{self.__class__.__name__}' with {len(self)} capsules"

    def to_dict(self) -> dict:
        """
        Returns metadata for this MicArray as a dictionary.
        """
        # Try and get all coordinate types for this microphone array
        coords = ["coordinates_absolute", "coordinates_polar", "coordinates_center", "coordinates_cartesian"]
        coord_dict = {}
        for coord_type in coords:
            try:
                coord_val = getattr(self, coord_type)
            # Skip over cases where this microphone doesn't have
            except NotImplementedError:
                coord_val = None
            # Need to parse arrays to list so that JSON serialising will work OK later on
            else:
                if isinstance(coord_val, np.ndarray):
                    coord_val = coord_val.tolist()
            coord_dict[coord_type] = coord_val
        return dict(
            name=self.name,
            is_spherical=self.is_spherical,
            n_capsules=self.n_capsules,
            capsule_names=self.capsule_names,
            **coord_dict
        )


@dataclass(repr=False)
class MonoCapsule(MicArray):
    """
    A single mono microphone capsule
    """
    name: str = "monocapsule"
    is_spherical: bool = False

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        return np.array([[0., 0., 0.]])

    @property
    def capsule_names(self) -> list[str]:
        return ["mono"]


@dataclass(repr=False)
class AmbeoVR(MicArray):
    """
    Sennheiser AmbeoVR microphone.

    Adapted from https://github.com/micarraylib/micarraylib/blob/main/micarraylib/arraycoords/array_shapes_raw.py
    """

    name: str = "ambeovr"
    is_spherical: bool = True

    @property
    def coordinates_polar(self) -> np.ndarray:
        return np.array([
            [45, 55, 0.01],
            [315, 125, 0.01],
            [135, 125, 0.01],
            [225, 55, 0.01]
        ])

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        """The positions of the capsules in Cartesian coordinates, i.e. as meters from the center of the array."""
        return utils.polar_to_cartesian(self.coordinates_polar)
    
    @property
    def capsule_names(self) -> list[str]:
        return ["FLU", "FRD", "BLD", "BRU"]


@dataclass(repr=False)
class Eigenmike32(MicArray):
    """
    Eigenmike 32 microphone.

    Adapted from https://eigenmike.com/sites/default/files/documentation-2023-10/EigenStudio%20User%20Manual%20R02D.pdf
    """

    name: str = "eigenmike32"
    is_spherical: bool = True

    @property
    def coordinates_polar(self) -> np.ndarray:
        # Adapted from Section 4.5 (pages 27--28) of official documentation
        return np.array([
            [0,    69, 0.042],
            [32,   90, 0.042],
            [0,   111, 0.042],
            [328,  90, 0.042],
            [0,    32, 0.042],
            [45,   55, 0.042],
            [69,   90, 0.042],
            [45,  125, 0.042],
            [0,   148, 0.042],
            [315, 125, 0.042],
            [291,  90, 0.042],
            [315,  55, 0.042],
            [91,   21, 0.042],
            [90,   58, 0.042],
            [90,  121, 0.042],
            [89,  159, 0.042],
            [180,  69, 0.042],
            [212,  90, 0.042],
            [180, 111, 0.042],
            [148,  90, 0.042],
            [180,  32, 0.042],
            [225,  55, 0.042],
            [249,  90, 0.042],
            [225, 125, 0.042],
            [180, 148, 0.042],
            [135, 125, 0.042],
            [111,  90, 0.042],
            [135,  55, 0.042],
            [269,  21, 0.042],
            [270,  58, 0.042],
            [270, 122, 0.042],
            [271, 159, 0.042]
        ])

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        """The positions of the capsules in Cartesian coordinates, i.e. as meters from the center of the array."""
        return utils.polar_to_cartesian(self.coordinates_polar)
    
    @property
    def capsule_names(self) -> list[str]:
        return [str(i) for i in range(1, 33)]


@dataclass(repr=False)
class Eigenmike64(MicArray):
    """
    Eigenmike 64 microphone.

    Adapted from https://eigenmike.com/sites/default/files/documentation-2024-09/getting%20started%20Guide%20to%20em64%20and%20ES3%20R01H.pdf
    """

    name: str = "eigenmike64"
    is_spherical: bool = True

    @property
    def coordinates_polar(self) -> np.ndarray:
        # These coordinates are obtained from the official Eigenmike 64 documentation (pages 27--29, Table 1)
        # Values for theta/phi are rounded to nearest integer
        # We assume a radius of 4.2 cm given the stated diameter of 8.4 cm
        return np.array([
            # Theta, Phi, Mic Z
            [197, 17, 0.042],
            [116, 22, 0.042],
            [82, 42, 0.042],
            [313, 13, 0.042],
            [43, 23, 0.042],
            [47, 53, 0.042],
            [336, 38, 0.042],
            [15, 43, 0.042],
            [204, 46, 0.042],
            [207, 70, 0.042],
            [247, 33, 0.042],
            [234, 60, 0.042],
            [265, 56, 0.042],
            [100, 67, 0.042],
            [105, 93, 0.042],
            [121, 48, 0.042],
            [127, 78, 0.042],
            [148, 62, 0.042],
            [163, 38, 0.042],
            [179, 64, 0.042],
            [21, 70, 0.042],
            [26, 96, 0.042],
            [48, 81, 0.042],
            [56, 106, 0.042],
            [71, 68, 0.042],
            [78, 92, 0.042],
            [293, 40, 0.042],
            [291, 41, 0.042],
            [318, 59, 0.042],
            [334, 82, 0.042],
            [352, 63, 0.042],
            [0, 90, 0.042],
            [174, 138, 0.042],
            [213, 140, 0.042],
            [252, 135, 0.042],
            [151, 109, 0.042],
            [241, 108, 0.042],
            [293, 142, 0.042],
            [331, 111, 0.042],
            [61, 109, 0.042],
            [227, 115, 0.042],
            [234, 86, 0.042],
            [194, 116, 0.042],
            [210, 95, 0.042],
            [183, 90, 0.042],
            [164, 111, 0.042],
            [157, 75, 0.042],
            [139, 80, 0.042],
            [136, 102, 0.042],
            [102, 53, 0.042],
            [113, 28, 0.042],
            [83, 27, 0.042],
            [308, 55, 0.042],
            [309, 0, 0.042],
            [278, 24, 0.042],
            [283, 49, 0.042],
            [253, 36, 0.042],
            [260, 61, 0.042],
            [60, 46, 0.042],
            [14, 54, 0.042],
            [32, 35, 0.042],
            [334, 46, 0.042],
            [2, 26, 0.042],
            [335, 35, 0.042],
        ])

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        """The positions of the capsules in Cartesian coordinates, i.e. as meters from the center of the array."""
        return utils.polar_to_cartesian(self.coordinates_polar)
    
    @property
    def capsule_names(self) -> list[str]:
        return [str(i) for i in range(1, 65)]


# A list of all mic array objects
MICARRAY_LIST = [
    Eigenmike32,
    Eigenmike64,
    AmbeoVR,
    MonoCapsule
]


def sanitize_microphone_input(microphone_type: Any) -> Type['MicArray']:
    """
    Sanitizes any microphone input into the correct 'MicArray' class.

    Returns:
        Type['MicArray']: the sanitized microphone class, ready to be instantiated
    """

    # Parsing the microphone type
    # If None, get a random microphone and use a randomized position
    if microphone_type is None:
        logger.warning(f"No microphone type provided, using a mono microphone capsule in a random position!")
        # Get a random microphone class
        sanitized_microphone = MonoCapsule

    # If a string, use the desired microphone type but get a random position
    elif isinstance(microphone_type, str):
        sanitized_microphone = get_micarray_from_string(microphone_type)

    # If a class contained inside MICARRAY_LIST
    elif microphone_type in MICARRAY_LIST:
        sanitized_microphone = microphone_type

    # Otherwise, we don't know what the microphone is
    else:
        raise TypeError(f"Could not parse microphone type {type(microphone_type)}")

    return sanitized_microphone


def get_micarray_from_string(micarray_name: str) -> Type['MicArray']:
    """
    Given a string representation of a microphone array (e.g., `eigenmike32`), return the correct MicArray object
    """
    # These are the name attributes for all valid microphone arrays
    acceptable_values = [ma().name for ma in MICARRAY_LIST]
    if micarray_name not in acceptable_values:
        raise ValueError(f"Cannot find array {micarray_name}: expected one of {','.join(acceptable_values)}")
    else:
        return [ma for ma in MICARRAY_LIST if ma.name == micarray_name][0]
