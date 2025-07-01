#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements dataclasses for working with common microphone array types"""

from dataclasses import dataclass, field

import numpy as np

from audiblelight import utils

__all__ = [
    "get_micarray_from_string",
    "MicArray",
    "Eigenmike32",
    "Eigenmike64",
    "Eigenmike32Tetrahedral",
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
            return self._coordinates_absolute

    @property
    def coordinates_center(self) -> np.ndarray:
        if self._coordinates_center is None:
            raise NotImplementedError("Must call `.set_absolute_coordinates` first!")
        else:
            return self._coordinates_center

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
        self._coordinates_center = utils.coerce2d(mic_center)
        self._coordinates_absolute = self.coordinates_cartesian + self._coordinates_center
        return self._coordinates_absolute

    def __len__(self) -> int:
        return self.n_capsules

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('n_capsules'={self.n_capsules}, 'is_spherical'={self.is_spherical})"

    def __str__(self) -> str:
        return f"Microphone array '{self.__class__.__name__}' with {len(self)} capsules"


@dataclass
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


@dataclass
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


@dataclass
class Eigenmike32Tetrahedral(Eigenmike32):
    """
    Eigenmike 32 microphone in tetrahedral form, using only 4 capsules.

    Following [1], we use channels 6, 10, 22, 26 from the Eigenmike 32 to obtain a lower resolution, four channel
    array. This can be useful, for instance, in training a model to upsample from the lower resolution input to the
    original, 32-channel input.

    [1] S. Adavanne, A. Politis, and T. Virtanen, "Localization, detection and tracking of multiple moving sound
    sources with a convolutional recurrent neural network" in Acoustic Scenes and Events 2019 Workshop (DCASE2019),
    2019, p. 20.
    """

    name: str = "eigenmike32_tetrahedral"
    is_spherical: bool = True

    @property
    def coordinates_polar(self) -> np.ndarray:
        return super().coordinates_polar[[6, 10, 22, 26], :]

    @property
    def capsule_names(self) -> list[str]:
        return [str(i) for i in [6, 10, 22, 26]]


@dataclass
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
    Eigenmike32Tetrahedral,
    AmbeoVR,
]


def get_micarray_from_string(micarray_name: str) -> type[MicArray]:
    """Given a string representation of a microphone array (e.g., `eigenmike32`), return the correct MicArray object"""
    # These are the name attributes for all valid microphone arrays
    acceptable_values = [ma().name for ma in MICARRAY_LIST]
    if micarray_name not in acceptable_values:
        raise ValueError(f"Cannot find array {micarray_name}: expected one of {','.join(acceptable_values)}")
    else:
        return [ma for ma in MICARRAY_LIST if ma.name == micarray_name][0]
