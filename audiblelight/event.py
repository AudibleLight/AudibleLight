#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for placing audio files within Space and Scene objects"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import soundfile as sf

from audiblelight.worldstate import Emitter
from audiblelight import utils


# class Event_:
#     """
#     Represents a single audio event taking place inside a Scene.
#     """
#     def __init__(
#             self,
#             filepath: Union[str, Path],
#             start_timepoint: float,
#             duration: float,
#             snr: float,
#             emitters: list[Emitter],
#             spatial_resolution: Optional = None,
#             spatial_velocity: Optional = None,
#             class_id: Optional[int] = None,
#             label: Optional[str] = "",
#     ):
#         pass


class Event:
    """
    Represents a single audio event taking place inside a Scene.
    """
    def __init__(
            self,
            filepath: Union[str, Path],
            start_timepoint: float,
            duration: float,
            snr: float,
            start_coordinates_absolute: np.ndarray,
            start_coordinates_relative: dict[str, np.ndarray],
            sample_rate: Optional[int] = utils.SAMPLE_RATE,
            is_moving: Optional[bool] = False,
            end_coordinates_absolute: Optional[np.ndarray] = None,
            end_coordinates_relative: Optional[dict[str, np.ndarray]] = None,
            spatial_resolution: Optional = None,
            spatial_velocity: Optional = None,
            class_id: Optional[int] = None,
            label: Optional[str] = "",
    ):
        """
        Initializes the Event object, representing a single audio event taking place inside a Scene.

        Args:
            filepath: Path to the audio file.
            start_timepoint: Time in seconds within the Scene.
            duration: Event duration in seconds
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            start_coordinates_absolute: Position of the Event within the Space in absolute Cartesian coordinates
            start_coordinates_relative: Cartesian coordinates of the Event with respect to the microphones in the Space
            is_moving (optional): Whether the Event is moving or not, defaults to False (i.e., static event)

        """
        # Setting attributes for audio
        self.filepath = utils.sanitise_filepath(filepath)    # will raise an error if not found on disk
        self.audio = None    # will be loaded when calling `load_audio` for the first time
        self._audio_loaded = False
        self.snr = snr
        self.sample_rate = utils.sanitise_positive_number(sample_rate)
        # Metadata attributes
        self.filename = self.filepath.name
        self.class_id = class_id
        self.label = label

        # Setting start, duration times
        #  The validation functions raise an error if non-numeric or not positive
        self.start_timepoint = utils.sanitise_positive_number(start_timepoint)
        self.duration = utils.sanitise_positive_number(duration)
        self.end_timepoint = self.start_timepoint + self.duration

        # Set start coordinates in absolute terms
        #  These should already be sanitised to be located within the bounds of the mesh, be an acceptable distance
        #  from microphone objects, etc., so the only thing we validate here is that they are the correct shape and type
        self.start_coordinates_absolute = utils.sanitise_coordinates(start_coordinates_absolute)
        self.start_coordinates_relative_cartesian = dict()
        self.start_coordinates_relative_polar = dict()
        # Iterate through the relative coordinates
        #  This is a dictionary of microphone aliases and coordinates
        for coords_alias, coords in start_coordinates_relative:
            coords = utils.sanitise_coordinates(coords)
            self.start_coordinates_relative_cartesian[coords_alias] = coords
            self.start_coordinates_relative_polar[coords_alias] = utils.cartesian_to_polar(coords)

        # Handling moving source attributes
        assert isinstance(is_moving, bool), "Expected `is_moving` to be of type `bool`"
        self.is_moving = is_moving
        # If the source is moving, we should provide some of these optional attributes
        if self.is_moving:
            expected_attrs = [end_coordinates_relative, end_coordinates_absolute, spatial_resolution, spatial_velocity]
            assert all([at is not None for at in expected_attrs])
            # Set the attributes
            self.spatial_resolution = spatial_resolution
            self.spatial_velocity = utils.sanitise_positive_number(spatial_velocity)
            # Update all the coordinates
            self.end_coordinates_absolute = utils.sanitise_coordinates(end_coordinates_absolute)
            self.end_coordinates_relative_cartesian = dict()
            self.end_coordinates_relative_polar = dict()
            for coords_alias, coords in start_coordinates_relative:
                coords = utils.sanitise_coordinates(coords)
                self.end_coordinates_relative_cartesian[coords_alias] = coords
                self.end_coordinates_relative_polar[coords_alias] = utils.cartesian_to_polar(coords)
        # Otherwise, if the source is not moving, we don't need to set these attributes
        else:
            self.spatial_resolution = None
            self.spatial_velocity = None
            self.end_coordinates_absolute = None
            self.end_coordinates_relative_cartesian = None
            self.end_coordinates_relative_polar = None

    def load_audio(self) -> np.ndarray:
        """
        Returns the audio array of the Event.
        """
        # If we've already loaded the audio, it should be cached and we can return straight away
        if self._audio_loaded and isinstance(self.audio, np.ndarray):
            return self.audio
        # Otherwise, we need to load up the audio
        else:
            self.audio, _ = sf.read(str(self.filepath), samplerate=self.sample_rate, always_2d=True,)
            self._audio_loaded = True
            return self.audio

    def to_dict(self) -> dict:
        """
        Returns metadata for this Event as a dictionary.
        """
        return dict(
            filename=self.filename,
            filepath=self.filepath,
            class_id=self.class_id,
            label=self.label,
            start_timepoint=self.start_timepoint,
            end_timepoint=self.end_timepoint,
            duration=self.duration,
            snr=self.snr,
            start_coordinates_absolute=self.start_coordinates_absolute,
            start_coordinates_relative_cartesian=self.start_coordinates_relative_cartesian,
            start_coordinates_relative_polar=self.start_coordinates_relative_polar,
            spatial_resolution=self.spatial_resolution,
            spatial_velocity=self.spatial_velocity,
            end_coordinates_absolute=self.end_coordinates_absolute,
            end_coordinates_relative_cartesian=self.end_coordinates_relative_cartesian,
            end_coordinates_relative_polar=self.end_coordinates_relative_polar,
        )
