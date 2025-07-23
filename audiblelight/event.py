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
            scene_start: Optional[float] = None,
            event_start: Optional[float] = None,
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
        self.snr = snr
        self.sample_rate = utils.sanitise_positive_number(sample_rate)
        self.alias = alias

        # Spatial audio attributes, set in the synthesizer
        self.spatial_audio = None

        # Spatial attributes
        self.spatial_resolution = spatial_resolution
        self.spatial_velocity = spatial_velocity

        # Metadata attributes
        self.filename = self.filepath.name
        #  Attempt to infer class ID and labels in cases where only one is provided
        self.class_id, self.class_label = _infer_dcase_class_id_labels(class_id, class_label)

        # Get the full duration of the audio file
        self.audio_full_duration = utils.sanitise_positive_number(librosa.get_duration(path=self.filepath))
        # Event start is the offset from the start of the audio file
        self.event_start = self._parse_audio_start(event_start)
        # Scene start is the time the event starts in the scene
        self.scene_start = utils.sanitise_positive_number(scene_start) if scene_start is not None else 0.
        # Safely parse the duration of the audio file with an optional override
        self.duration = self._parse_duration(duration)

        # Now we can safely get the ending time of the event
        self.event_end = self.event_start + self.duration
        self.scene_end = self.scene_start + self.duration

        # List of emitter objects associated with this event
        if isinstance(emitters, Emitter):
            emitters = [emitters]    # pad to a list
        elif isinstance(emitters, list):
            assert all(isinstance(em, Emitter) for em in emitters), "All objects must be of `Emitter` type"
            assert len(emitters) >= 1, "At least one emitter must be provided"
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

    @property
    def is_audio_loaded(self) -> bool:
        """
        Returns True if audio is loaded and valid (see `librosa.util.valid_audio` for more detail).
        """
        return self.audio is not None and librosa.util.valid_audio(self.audio)

    def _parse_audio_start(self, audio_start: Optional[utils.Numeric]) -> float:
        """
        Safely handle getting the start/offset time for an audio event, with an optional override.
        """
        if audio_start is None:
            event_start_ = 0.
        # Raise a warning and revert to 0 seconds when passed start time exceeds total duration of the audio file
        elif audio_start > self.audio_full_duration:
            logger.warning(f"Event start time ({audio_start:.2f} seconds) exceeds duration of the audio file "
                           f"({self.audio_full_duration:.2f} seconds). Start time will be set to 0.")
            event_start_ = 0.
        else:
            event_start_ = audio_start
        return utils.sanitise_positive_number(event_start_)

    def _parse_duration(self, duration: Optional[float]) -> float:
        """
        Safely handle getting the duration of an audio file, with an optional override.
        """
        # If we haven't passed in an override, just use the full duration of the audio, minus the offset
        if duration is None:
            return utils.sanitise_positive_number(self.audio_full_duration - self.event_start)
        else:
            # Otherwise, check that our duration is valid
            duration = utils.sanitise_positive_number(duration)
            # If the duration combined with the offset time is longer than the actual audio itself
            if self.event_start + duration > self.audio_full_duration:
                logger.warning(
                    f"Duration {duration:.2f} is longer than audio duration {self.audio_full_duration:.2f} with "
                    f"given audio start time {self.event_start:.2f}. Falling back to using full audio duration."
                )
                # Fall back to using
                return self.audio_full_duration - self.event_start
            else:
                return duration

    # noinspection PyTypeChecker
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
        # If we've already loaded the audio, and it is still valid, we can return it straight away
        if self.is_audio_loaded and not ignore_cache:
            return self.audio

        else:
            # Otherwise, we need to load up the audio
            #  Using librosa, this will resample to the desired sample rate, convert to mono, set the offset to the
            #  desired event start time, and trim the duration to the desired duration
            self.audio, _ = librosa.load(
                self.filepath,
                sr=self.sample_rate,
                mono=True,
                offset=self.event_start,
                duration=self.duration,
                dtype=np.float32
            )

        return self.audio

    def to_dict(self) -> dict:
        """
        Returns metadata for this Event as a dictionary.
        """
        def coerce(inp):
            if isinstance(inp, dict):
                return {k: coerce(v) for k, v in inp.items()} if inp else None
            elif isinstance(inp, np.ndarray):
                return inp.tolist()
            else:
                return inp

        return dict(
            # Metadata
            alias=self.alias,
            filename=str(self.filename),
            filepath=str(self.filepath),
            class_id=self.class_id,
            class_label=self.class_label,
            # Audio stuff
            scene_start=self.scene_start,
            scene_end = self.scene_end,
            event_start=self.event_start,
            event_end=self.event_end,
            duration=self.duration,
            snr=self.snr,
            # Spatial stuff (inherited from Emitter objects)
            spatial_resolution=self.spatial_resolution,
            spatial_velocity=self.spatial_velocity,
            start_coordinates=dict(
                absolute=coerce(self.start_coordinates_absolute),
                relative_cartesian=coerce(self.start_coordinates_relative_cartesian),
                relative_polar=coerce(self.start_coordinates_relative_polar),
            ),
            end_coordinates=dict(
                absolute=coerce(self.end_coordinates_absolute),
                relative_cartesian=coerce(self.end_coordinates_relative_cartesian),
                relative_polar=coerce(self.end_coordinates_relative_polar),
            )
        )
