#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for placing audio files within Space and Scene objects"""

from pathlib import Path
from typing import Union, Optional

import librosa
import numpy as np
from loguru import logger

from audiblelight.worldstate import Emitter
from audiblelight import utils

# Mapping from sound events to labels used in DCASE challenge
DCASE_SOUND_EVENT_CLASSES = {
    "femaleSpeech": 0,
    "maleSpeech": 1,
    "clapping": 2,
    "telephone": 3,
    "laughter": 4,
    "domesticSounds": 5,
    "footsteps": 6,
    "doorCupboard": 7,
    "music": 8,
    "musicInstrument": 9,
    "waterTap": 10,
    "bell": 11,
    "knock": 12,
}
_DCASE_SOUND_EVENT_CLASSES_INV = {v: k for v, k in DCASE_SOUND_EVENT_CLASSES.items()}


def _infer_dcase_class_id_labels(
    class_id: Optional[int],
    class_label: Optional[str]
) -> tuple[Optional[int], Optional[str]]:
    """
    Infers missing DCASE class ID or label if only one is provided.

    - If only class_label is provided, class_id is inferred.
    - If only class_id is provided, class_label is inferred.
    - If both are provided or both are None, returns them as-is.
    """
    if class_id is None and class_label is not None:
        class_id = DCASE_SOUND_EVENT_CLASSES.get(class_label)

    elif class_id is not None and class_label is None:
        class_label = _DCASE_SOUND_EVENT_CLASSES_INV.get(class_id)

    return class_id, class_label


class Event:
    """
    Represents a single audio event taking place inside a Scene.
    """

    def __init__(
            self,
            filepath: Union[str, Path],
            alias: str,
            emitters: list[Emitter],
            scene_start: Optional[float] = 0.,
            event_start: Optional[float] = 0.,
            duration: Optional[float] = None,
            snr: Optional[float] = None,
            sample_rate: Optional[int] = utils.SAMPLE_RATE,
            class_id: Optional[int] = None,
            class_label: Optional[str] = None,
            spatial_resolution: Optional[Union[int, float]] = None,
            spatial_velocity: Optional[Union[int, float]] = None,
    ):
        """
        Initializes the Event object, representing a single audio event taking place inside a Scene.

        Args:
            filepath: Path to the audio file.
            alias: Label to refer to this Event by inside the Scene
            emitters: List of Emitter objects associated with this event.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
                If not provided, defaults to the beginning of the Scene (i.e., 0 seconds).
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
                If not provided, defaults to starting the audio at the very beginning (i.e., 0 seconds).
            duration: Time the Event audio lasts in seconds. Must be a positive number.
                If None or greater than the duration of the audio, defaults to using the full duration of the audio.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            sample_rate: If not None, the audio will be resampled to the given sample rate.
            class_label: Optional label to use for sound event class.
                If not provided, the label will attempt to be inferred from the ID using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, the ID will attempt to be inferred from the label using the DCASE sound event classes.
        """
        # Setting attributes for audio
        self.filepath = utils.sanitise_filepath(filepath)    # will raise an error if not found on disk, coerces to Path
        self.audio = None    # will be loaded when calling `load_audio` for the first time
        self._audio_loaded = False
        self.snr = snr
        self.sample_rate = utils.sanitise_positive_number(sample_rate)
        self.alias = alias

        # Spatial attributes
        self.spatial_resolution = spatial_resolution
        self.spatial_velocity = spatial_velocity

        # Metadata attributes
        self.filename = self.filepath.name
        #  Attempt to infer class ID and labels in cases where only one is provided
        self.class_id, self.class_label = _infer_dcase_class_id_labels(class_id, class_label)

        # Setting start, duration times
        self._audio_full_duration = librosa.get_duration(path=self.filepath)
        #  The validation functions raise an error if non-numeric or not positive
        self.scene_start = utils.sanitise_positive_number(scene_start) if scene_start is not None else scene_start
        self.event_start = utils.sanitise_positive_number(event_start) if event_start is not None else event_start
        if duration is not None:
            duration = utils.sanitise_positive_number(duration)
            if duration > self._audio_full_duration:
                logger.warning(
                    f"Duration {duration:.2f} is longer than audio duration {self._audio_full_duration:.2f}. "
                    f"Falling back to using audio duration."
                )
                duration = self._audio_full_duration
        self.duration = duration


        # List of emitter objects associated with this event
        if isinstance(emitters, Emitter):
            emitters = [emitters]    # pad to a list
        assert all(isinstance(em, Emitter) for em in emitters)
        self.emitters = emitters
        #  If more than one emitter, the sound source is moving; if only one emitter, the sound source is stationary
        self.is_moving = len(self.emitters) > 1

        # We presume that the list of emitter objects is "sorted"
        #  i.e., that the first emitter corresponds to the start position and the last to the end
        self.start_coordinates_absolute = self.emitters[0].coordinates_absolute
        self.start_coordinates_relative_cartesian = self.emitters[0].coordinates_relative_cartesian
        self.start_coordinates_relative_polar = self.emitters[0].coordinates_relative_polar

        # Set the ending coordinates: if the object is not moving, these are the same as the starting coordinates.
        if self.is_moving:
            self.end_coordinates_absolute = self.emitters[-1].coordinates_absolute
            self.end_coordinates_relative_cartesian = self.emitters[-1].coordinates_relative_cartesian
            self.end_coordinates_relative_polar = self.emitters[-1].coordinates_relative_polar
        else:
            self.end_coordinates_absolute = self.start_coordinates_absolute
            self.end_coordinates_relative_cartesian = self.start_coordinates_relative_cartesian
            self.end_coordinates_relative_polar = self.start_coordinates_relative_polar

    def load_audio(self, ignore_cache: bool = False) -> np.ndarray:
        """
        Returns the audio array of the Event.

        The audio will be loaded, resampled to the desired sample rate, converted to mono, and then truncated to match
        the event start time and duration.

        After calling this function once, `audio` is cached as an attribute of this Event instance, and this
        attribute will be returned on successive calls unless `ignore_cache` is True.

        Returns:
            np.ndarray: the audio array.
        """
        # Invalidate the cache if required
        if ignore_cache:
            self._audio_loaded = False
            self.audio = None

        # If we've already loaded the audio, it should be cached and we can return straight away
        if self._audio_loaded and isinstance(self.audio, np.ndarray):
            return self.audio

        else:
            # Otherwise, we need to load up the audio
            #  Using soundfile, this will resample to the desired sample rate.
            audio, _ = librosa.load(self.filepath, sr=self.sample_rate)

            # Convert the audio file to mono
            #  By passing `always_2d` above, we ensure that the audio is always a 2D array regardless of if the
            #  actual audio is mono or stereo (etc.), which standardises the number of dimensions
            audio = utils.audio_to_mono(audio)

            # Truncate the audio to the given start time and audio duration
            audio = utils.truncate_audio(audio, self.sample_rate, self.event_start, self.duration)

            # Set the attributes correctly
            self.audio = audio
            self._audio_loaded = True

        return self.audio

    def to_dict(self) -> dict:
        """
        Returns metadata for this Event as a dictionary.
        """
        return dict(
            alias=self.alias,
            filename=self.filename,
            filepath=self.filepath,
            class_id=self.class_id,
            class_label=self.class_label,
            scene_start=self.scene_start,
            event_start=self.event_start,
            duration=self.duration,
            # TODO: we need start and end timepoints for the scene + event audio
            # end_timepoint=self.scene_end,
            snr=self.snr,
            spatial_resolution=self.spatial_resolution,
            spatial_velocity=self.spatial_velocity,
            start_coordinates=dict(
                absolute=self.start_coordinates_absolute,
                relative_cartesian=self.start_coordinates_relative_cartesian,
                relative_polar=self.start_coordinates_relative_polar,
            ),
            end_coordinates=dict(
                absolute=self.end_coordinates_absolute,
                relative_cartesian=self.end_coordinates_relative_cartesian,
                relative_polar=self.end_coordinates_relative_polar,
            )
        )
