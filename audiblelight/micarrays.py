#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements dataclasses for working with common microphone array types"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Type

import numpy as np
from deepdiff import DeepDiff
from loguru import logger
from rlr_audio_propagation import ChannelLayout, ChannelLayoutType

from audiblelight import utils

__all__ = [
    "sanitize_microphone_input",
    "MicArray",
    "Eigenmike32",
    "Eigenmike64",
    "MonoCapsule",
    "AmbeoVR",
    "MICARRAY_LIST",
    "FOAListener",
]


@dataclass(eq=False)
class MicArray:
    """
    This is the base class for all microphone array types.

    Attributes:
        name (str): the name of the array.
        is_spherical (bool): whether the array is spherical. If False, positions_spherical will be None.
        channel_layout_type (str): the expected channel layout for each capsule. If "mic" (default), one channel will
            be created for every capsule. If "foa", four channels will be created per capsule.

    Properties:
        coordinates_polar (np.array): the positions of the capsules on the array, given as azimuth, elevation, radius.
            Azimuth is measured counter-clockwise in degrees between 0 and 360, where 0 == the front of the microphone.
            Elevation is measured between -90 and 90 degrees, where 0 == "straight ahead", 90 == "up", -90 == "down".
            Radius is measured in meters away from the center of the array.
            Note that, when `is_spherical` is False, this property will be None.
        coordinates_cartesian (np.array): the positions of the capsules in Cartesian (XYZ) coordinates, with distance
            measured using meters away from the center of the array
        coordinates_absolute (np.array): the absolute position of all capsules based on a provided center.
        n_capsules (int): number of capsules in the array
        capsule_names (list[str]): the names of the microphone capsules
    """

    name: str = ""
    is_spherical: bool = False
    channel_layout_type: str = "mic"

    irs: np.ndarray = field(default=None, init=False, repr=False)
    _coordinates_absolute: np.ndarray = field(default=None, init=False, repr=False)
    _coordinates_center: np.ndarray = field(default=None, init=False, repr=False)

    @property
    def channel_layout(self) -> ChannelLayout:
        """
        Returns the ray-tracing engine ChannelLayout object for this MicArray
        """
        if self.channel_layout_type == "mic":
            layout_type = ChannelLayoutType.Mono
            return ChannelLayout(layout_type, 1)
        elif self.channel_layout_type == "foa":
            layout_type = ChannelLayoutType.Ambisonics
            return ChannelLayout(layout_type, 4)
        elif self.channel_layout_type == "binaural":
            layout_type = ChannelLayoutType.Binaural
            return ChannelLayout(layout_type, 2)
        else:
            raise ValueError(
                f"Expected `channel_layout_type` to be one of 'mono', 'foa', 'binaural' "
                f"but got '{self.channel_layout_type}'"
            )
        # return ChannelLayout(layout_type, self.n_capsules)

    @property
    def n_listeners(self) -> int:
        """
        Returns the number of listeners this `MicArray` should be associated with in the engine.

        If channel_layout == foa, we will only have 1 listener, but 4 "capsules" and IRs.
        Otherwise, if channel_layout == mono, we will have as many listeners as capsules.
        """
        if self.channel_layout_type == "mic":
            return self.n_capsules
        elif self.channel_layout_type == "foa":
            return 1
        elif self.channel_layout_type == "binaural":
            return 1
        else:
            raise ValueError(
                f"Expected `channel_layout_type` to be one of 'mono', 'foa', 'binaural' "
                f"but got '{self.channel_layout_type}'"
            )

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
        self._coordinates_absolute = self.coordinates_cartesian + utils.coerce2d(
            self._coordinates_center
        )
        return self._coordinates_absolute

    def __len__(self) -> int:
        """
        Return the number of capsules associated with this microphone
        """
        return self.n_capsules

    def __repr__(self) -> str:
        """
        Return a JSON-formatted string representation of this microphone array
        """
        return utils.repr_as_json(self)

    def __str__(self) -> str:
        """
        Return a string representation of this microphone array
        """
        return f"Microphone array '{self.__class__.__name__}' with {len(self)} capsules"

    def __eq__(self, other: Any) -> bool:
        """
        Compare two MicArray objects for equality.

        Returns:
            bool: True if the MicArray objects are identical, False otherwise
        """

        # Non-MicArray objects are always not equal
        if not isinstance(other, MicArray):
            return False

        # We use dictionaries to compare both objects together
        d1 = self.to_dict()
        d2 = other.to_dict()

        # Compute the deepdiff between both dictionaries
        diff = DeepDiff(
            d1,
            d2,
            ignore_order=True,
            significant_digits=4,
            ignore_numeric_type_changes=True,
        )

        # If there is no difference, there should be no keys in the deepdiff object
        return len(diff) == 0

    def to_dict(self) -> dict:
        """
        Returns metadata for this MicArray as a dictionary.
        """
        # Try and get all coordinate types for this microphone array
        coords = [
            "coordinates_absolute",
            # "coordinates_polar",
            "coordinates_center",
            # "coordinates_cartesian",
        ]
        coord_dict = OrderedDict()
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
            micarray_type=self.__class__.__name__,
            is_spherical=self.is_spherical,
            channel_layout_type=self.channel_layout_type,
            n_capsules=self.n_capsules,
            capsule_names=self.capsule_names,
            **coord_dict,
        )

    @staticmethod
    def _get_mic_class(input_dict: dict[str, Any]) -> Type["MicArray"]:
        """
        Given a dictionary, get the desired MicArray class.

        Arguments:
            input_dict (dict[str, Any]): dictionary to instantiate MicArray class

        Returns:
            MicArray object
        """
        # Get the class type of the desired microphone
        desired_mic = input_dict.pop("micarray_type", "mic")
        if desired_mic not in MICARRAY_CLASS_MAPPING:
            raise ValueError(
                f"{desired_mic} is not a valid microphone array type! "
                f"Expected one of {', '.join(MICARRAY_CLASS_MAPPING.keys())}"
            )
        # Instantiate the microphone and set its coordinates
        return MICARRAY_CLASS_MAPPING[desired_mic]

    def _set_attribute(self, attr_name: str, value: Any) -> None:
        """
        Set an attribute on the MicArray object.

        Arguments:
            attr_name (str): name of the attribute to set
            value (Any): value to set

        Returns:
            None

        Raises:
            AttributeError: if the value for an attribute does not match the default, when already set
        """
        # "de-serialise" lists back to arrays, ignoring strings
        if isinstance(value, list) and not isinstance(value[0], str):
            value = np.asarray(value)

        # Try and set the attribute
        try:
            hasat = hasattr(
                self, attr_name
            )  # need to be defensive, can sometimes hit NotImplementedError
        except NotImplementedError:
            return

        if hasat:
            try:
                setattr(self, attr_name, value)

            # We can't always set some dataclass attributes, but we should check that the default value is
            #  equivalent to what has been passed in
            except AttributeError:
                expected = getattr(self, attr_name)
                # Need to use different equality comparisons for arrays vs non-arrays
                eq = (
                    np.isclose(expected, value, atol=utils.SMALL).all()
                    if isinstance(value, np.ndarray)
                    else expected == value
                )

                if not eq:
                    raise AttributeError(
                        f"Expected attribute {attr_name} to have value {expected}, but got {value}!"
                    )
                else:
                    return

        # Ignore cases where there is no attribute
        else:
            return

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate a `MicArray` from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `MicArray`.

        Returns:
            MicArray instance.
        """
        mic_class = cls._get_mic_class(input_dict)()
        mic_class.set_absolute_coordinates(input_dict["coordinates_center"])

        # Set any other valid parameters for the microphone as well
        for k, v in input_dict.items():
            mic_class._set_attribute(k, v)

        return mic_class


@dataclass(repr=False, eq=False)
class MonoCapsule(MicArray):
    """
    A single mono microphone capsule
    """

    name: str = "monocapsule"
    is_spherical: bool = False
    channel_layout_type: str = "mic"

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0]])

    @property
    def capsule_names(self) -> list[str]:
        return ["mono"]


@dataclass(repr=False, eq=False)
class FOAListener(MicArray):
    """
    First Order Ambisonics (FOA) microphone "capsule"

    This implementation uses a single listener with 4 ambisonics channels (W, X, Y, Z)
    following the AmbiX convention. Unlike `AmbeoVR` which places 4 separate mono capsules,
    this represents a single point in space with 4-channel ambisonics encoding.
    """

    name: str = "foalistener"
    is_spherical: bool = False
    channel_layout_type: str = "foa"

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        # Note that this just means there is a single capsule placed at the same position as the
        #  "origin" of the microphone. Normally, this array would return the cartesian coords
        #  of the capsules WRT the origin. However, as there is only one """capsule""" here,
        #  the array just contains zeroes.
        return np.array([[0.0, 0.0, 0.0]])

    @property
    def capsule_names(self) -> list[str]:
        return ["w", "x", "y", "z"]


@dataclass(repr=False, eq=False)
class AmbeoVR(MicArray):
    """
    Sennheiser AmbeoVR microphone.

    Adapted from https://github.com/micarraylib/micarraylib/blob/main/micarraylib/arraycoords/array_shapes_raw.py
    """

    name: str = "ambeovr"
    is_spherical: bool = True
    channel_layout_type: str = "mic"

    @property
    def coordinates_polar(self) -> np.ndarray:
        return np.array(
            [[45, 35, 0.01], [-45, -35, 0.01], [135, -35, 0.01], [-135, 35, 0.01]]
        )

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        """The positions of the capsules in Cartesian coordinates, i.e. as meters from the center of the array."""
        return utils.polar_to_cartesian(self.coordinates_polar)

    @property
    def capsule_names(self) -> list[str]:
        return ["FLU", "FRD", "BLD", "BRU"]


@dataclass(repr=False, eq=False)
class Eigenmike32(MicArray):
    """
    Eigenmike 32 microphone.

    Adapted from https://eigenmike.com/sites/default/files/documentation-2023-10/EigenStudio%20User%20Manual%20R02D.pdf
    """

    name: str = "eigenmike32"
    is_spherical: bool = True
    channel_layout_type: str = "mic"

    @property
    def coordinates_polar(self) -> np.ndarray:
        # Adapted from Section 4.5 (pages 27--28) of official documentation
        return np.array(
            [
                [0.0, 21.0, 0.042],
                [32.0, 0.0, 0.042],
                [0.0, -21.0, 0.042],
                [-32.0, 0.0, 0.042],
                [0.0, 58.0, 0.042],
                [45.0, 35.0, 0.042],
                [69.0, 0.0, 0.042],
                [45.0, -35.0, 0.042],
                [0.0, -58.0, 0.042],
                [-45.0, -35.0, 0.042],
                [-69.0, 0.0, 0.042],
                [-45.0, 35.0, 0.042],
                [91.0, 69.0, 0.042],
                [90.0, 32.0, 0.042],
                [90.0, -31.0, 0.042],
                [89.0, -69.0, 0.042],
                [180.0, 21.0, 0.042],
                [-148.0, 0.0, 0.042],
                [180.0, -21.0, 0.042],
                [148.0, 0.0, 0.042],
                [180.0, 58.0, 0.042],
                [-135.0, 35.0, 0.042],
                [-111.0, 0.0, 0.042],
                [-135.0, -35.0, 0.042],
                [180.0, -58.0, 0.042],
                [135.0, -35.0, 0.042],
                [111.0, 0.0, 0.042],
                [135.0, 35.0, 0.042],
                [-91.0, 69.0, 0.042],
                [-90.0, 32.0, 0.042],
                [-90.0, -32.0, 0.042],
                [-89.0, -69.0, 0.042],
            ]
        )

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        """The positions of the capsules in Cartesian coordinates, i.e. as meters from the center of the array."""
        return utils.polar_to_cartesian(self.coordinates_polar)

    @property
    def capsule_names(self) -> list[str]:
        return [str(i) for i in range(1, 33)]


@dataclass(repr=False, eq=False)
class Eigenmike64(MicArray):
    """
    Eigenmike 64 microphone.

    Adapted from https://eigenmike.com/sites/default/files/documentation-2024-09/getting%20started%20Guide%20to%20em64%20and%20ES3%20R01H.pdf
    """

    name: str = "eigenmike64"
    is_spherical: bool = True
    channel_layout_type: str = "mic"

    @property
    def coordinates_polar(self) -> np.ndarray:
        # These coordinates are obtained from the official Eigenmike 64 documentation (pages 27--29, Table 1)
        # Values for theta/phi are rounded to nearest integer
        # We assume a radius of 4.2 cm given the stated diameter of 8.4 cm
        return np.array(
            [
                [-163.0, 73.0, 0.042],
                [116.0, 68.0, 0.042],
                [82.0, 48.0, 0.042],
                [-47.0, 77.0, 0.042],
                [43.0, 67.0, 0.042],
                [47.0, 37.0, 0.042],
                [-24.0, 52.0, 0.042],
                [15.0, 47.0, 0.042],
                [-156.0, 44.0, 0.042],
                [-153.0, 20.0, 0.042],
                [-113.0, 57.0, 0.042],
                [-126.0, 30.0, 0.042],
                [-95.0, 34.0, 0.042],
                [100.0, 23.0, 0.042],
                [105.0, -3.0, 0.042],
                [121.0, 42.0, 0.042],
                [127.0, 12.0, 0.042],
                [148.0, 28.0, 0.042],
                [163.0, 52.0, 0.042],
                [179.0, 26.0, 0.042],
                [21.0, 20.0, 0.042],
                [26.0, -6.0, 0.042],
                [48.0, 9.0, 0.042],
                [56.0, -16.0, 0.042],
                [71.0, 22.0, 0.042],
                [78.0, -2.0, 0.042],
                [-67.0, 50.0, 0.042],
                [-69.0, 49.0, 0.042],
                [-42.0, 31.0, 0.042],
                [-26.0, 8.0, 0.042],
                [-8.0, 27.0, 0.042],
                [0.0, 0.0, 0.042],
                [174.0, -48.0, 0.042],
                [-147.0, -50.0, 0.042],
                [-108.0, -45.0, 0.042],
                [151.0, -19.0, 0.042],
                [-119.0, -18.0, 0.042],
                [-67.0, -52.0, 0.042],
                [-29.0, -21.0, 0.042],
                [61.0, -19.0, 0.042],
                [-133.0, -25.0, 0.042],
                [-126.0, 4.0, 0.042],
                [-166.0, -26.0, 0.042],
                [-150.0, -5.0, 0.042],
                [-177.0, 0.0, 0.042],
                [164.0, -21.0, 0.042],
                [157.0, 15.0, 0.042],
                [139.0, 10.0, 0.042],
                [136.0, -12.0, 0.042],
                [102.0, 37.0, 0.042],
                [113.0, 62.0, 0.042],
                [83.0, 63.0, 0.042],
                [-52.0, 35.0, 0.042],
                [-51.0, 90.0, 0.042],
                [-82.0, 66.0, 0.042],
                [-77.0, 41.0, 0.042],
                [-107.0, 54.0, 0.042],
                [-100.0, 29.0, 0.042],
                [60.0, 44.0, 0.042],
                [14.0, 36.0, 0.042],
                [32.0, 55.0, 0.042],
                [-26.0, 44.0, 0.042],
                [2.0, 64.0, 0.042],
                [-25.0, 55.0, 0.042],
            ]
        )

    @property
    def coordinates_cartesian(self) -> np.ndarray:
        """The positions of the capsules in Cartesian coordinates, i.e. as meters from the center of the array."""
        return utils.polar_to_cartesian(self.coordinates_polar)

    @property
    def capsule_names(self) -> list[str]:
        return [str(i) for i in range(1, 65)]


# A list of all mic array objects
MICARRAY_LIST = [Eigenmike32, Eigenmike64, AmbeoVR, MonoCapsule, FOAListener]
MICARRAY_CLASS_MAPPING = {cls.__name__: cls for cls in MICARRAY_LIST}


def sanitize_microphone_input(microphone_type: Any) -> Type["MicArray"]:
    """
    Sanitizes any microphone input into the correct 'MicArray' class.

    Returns:
        Type['MicArray']: the sanitized microphone class, ready to be instantiated
    """

    # Parsing the microphone type
    # If None, get a random microphone and use a randomized position
    if microphone_type is None:
        logger.warning(
            "No microphone type provided, using a mono microphone capsule in a random position!"
        )
        # Get a random microphone class
        sanitized_microphone = MonoCapsule

    # If a string, use the desired microphone type but get a random position
    elif isinstance(microphone_type, str):
        sanitized_microphone = get_micarray_from_string(microphone_type)

    # If a class contained inside MICARRAY_LIST
    elif microphone_type in MICARRAY_LIST:
        sanitized_microphone = microphone_type

    # If an instance of a class contained inside MICARRAY_LIST
    elif type(microphone_type) in MICARRAY_LIST:
        sanitized_microphone = type(microphone_type)

    elif issubclass(microphone_type, MicArray):
        sanitized_microphone = microphone_type

    # Otherwise, we don't know what the microphone is
    else:
        raise TypeError(f"Could not parse microphone type {type(microphone_type)}")

    return sanitized_microphone


def get_micarray_from_string(micarray_name: str) -> Type["MicArray"]:
    """
    Given a string representation of a microphone array (e.g., `eigenmike32`), return the correct MicArray object
    """
    # These are the name attributes for all valid microphone arrays
    acceptable_values = [ma().name for ma in MICARRAY_LIST]
    if micarray_name not in acceptable_values:
        raise ValueError(
            f"Cannot find array {micarray_name}: expected one of {', '.join(acceptable_values)}"
        )
    else:
        # Using `next` avoids having to build the whole list
        return next(ma for ma in MICARRAY_LIST if ma.name == micarray_name)
