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
        elif self.channel_layout_type == "foa":
            layout_type = ChannelLayoutType.Ambisonics
        elif self.channel_layout_type == "binaural":
            layout_type = ChannelLayoutType.Binaural
        else:
            raise ValueError(
                f"Expected `channel_layout_type` to be one of 'mono', 'foa', 'binaural' "
                f"but got '{self.channel_layout_type}'"
            )
        return ChannelLayout(layout_type, self.n_capsules)

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
            [[-162.54386375754572, 73.23441392780798, 0.04199999898761863],
            [115.73396094280298, 68.03231814138955, 0.04200000076601955],
            [81.91098261318237, 47.60593590952149, 0.042000001086779926],
            [-46.640805068632254, 76.71834630962336, 0.041999998565651155],
            [43.178515531619084, 67.32717374455808, 0.04199999846745143],
            [46.7324101779103, 37.30751785094164, 0.04199999943071513],
            [-24.004232850134958, 52.19398953445288, 0.04199999882398202],
            [14.539804340340874, 46.60564155118599, 0.04199999936047616],
            [-155.5452811342395, 46.06137234773886, 0.041999998942159805],
            [-153.45795127452, 19.686795249056882, 0.04200000168943406],
            [-112.67807783770206, 56.776908272343654, 0.041999999231782055],
            [-126.18300363202042, 29.97430313827972, 0.042000000489402244],
            [-95.4562597627497, 33.52371479603167, 0.04200000263279551],
            [99.66685938458157, 22.50638786045734, 0.041999999560846005],
            [104.68416985113832, -3.273517013432528, 0.04199999961404582],
            [120.9227489702704, 41.57702282013416, 0.04199999937893374],
            [126.5130272611368, 11.92074908257109, 0.041999998960231426],
            [148.23676614789827, 27.931476898378715, 0.041999999917883536],
            [162.63807354134423, 51.28288227610622, 0.0419999998285492],
            [178.54978428624415, 26.19962426211601, 0.04200000071403506],
            [21.271478789044888, 19.805391529349816, 0.041999999604907925],
            [25.783359995115216, -6.245993285007529, 0.041999998951278115],
            [47.86068669401391, 8.900792583442001, 0.04199999856054826],
            [55.90745534029402, -16.09404469501315, 0.04199999954358831],
            [71.42852141661255, 22.246651893646902, 0.04200000169913675],
            [78.4921462664288, -1.706148023782233, 0.042000000080175064],
            [-66.77900947429889, 50.00152599303788, 0.04199999917880916],
            [-69.43167860622623, 21.227361373908828, 0.042000000639571336],
            [-41.86459524806503, 29.11312002545277, 0.0420000001440507],
            [-25.995838653098343, 7.716740510269007, 0.04199999751951004],
            [-7.977253310239233, 26.97526610478213, 0.041999998964521064],
            [0.0, 0.20597468141704964, 0.04199999939462087],
            [174.0334789982908, -47.516598049674734, 0.04200000168601247],
            [-147.2795074953714, -49.7603772780842, 0.042000001005051586],
            [-108.08212495056375, -45.21332577350694, 0.04199999970648593],
            [150.6470570354014, -70.36283201065848, 0.04199999986515194],
            [-119.17338368692005, -72.57696131461054, 0.04200000179684013],
            [-66.9375155999938, -52.06854078065325, 0.04199999917279667],
            [-28.990232224653116, -71.19865709297417, 0.04200000139983439],
            [60.82661409222707, -72.57696122568635, 0.041999998673038046],
            [-133.08650992975473, -25.536015842487487, 0.04199999851891228],
            [-126.07449416054621, 3.740581438490357, 0.04199999989655914],
            [-166.36179836787602, -26.01639288905614, 0.042000000287941736],
            [-150.33038190629227, -5.331279348380548, 0.041999999728892644],
            [-176.8310008077743, -0.06372861044180314, 0.04199999844939632],
            [163.71047395388496, -21.454870019220525, 0.04200000089384517],
            [156.95238514247103, 4.132934837615129, 0.042000001726650166],
            [139.43175462942096, -40.83984759287775, 0.041999999370817005],
            [135.97292383225542, -12.577542640372448, 0.04200000095671166],
            [102.32728603751016, -52.63746010427591, 0.04199999964418273],
            [112.5510842371462, -27.031976175723024, 0.04199999780290303],
            [83.14644691324017, -27.563053534325174, 0.04200000167373285],
            [-52.2922118442553, -25.888408136123594, 0.04200000013355441],
            [-50.86076906637201, 0.31001395792289016, 0.04200000158012028],
            [-81.74811223788424, -28.447823657053355, 0.04199999940819298],
            [-77.0264676522604, -3.933768556100541, 0.041999999529032724],
            [-106.8529804724323, -16.387494334159694, 0.04200000089562245],
            [-99.93121898173469, 8.948883473157895, 0.04200000081586629],
            [59.73935722453813, -45.97635639090507, 0.04200000076201796],
            [14.224107657250258, -52.67708465170158, 0.041999999704010284],
            [32.49010789047045, -30.655618697031322, 0.041999999252344815],
            [-25.92474142454067, -43.8834121825316, 0.04200000141063373],
            [2.08417870509163, -26.359072742802205, 0.04199999933685809],
            [-24.932349891430803, -17.46399045314746, 0.041999998617764274]]
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
