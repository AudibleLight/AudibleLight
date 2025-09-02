#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for placing audio files within Space and Scene objects"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Type, Union

import librosa
import numpy as np
from deepdiff import DeepDiff
from loguru import logger

from audiblelight import utils
from audiblelight.augmentation import EventAugmentation, validate_event_augmentation
from audiblelight.worldstate import Emitter

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

# If no duration override passed in, this will be the maximum duration for an event
MAX_DEFAULT_DURATION = 10


def infer_dcase_label_idx_from_filepath(
    filepath: Union[Path, str]
) -> Union[tuple[int, str], tuple[None, None]]:
    """
    Given a filepath, infer the DCASE class label and index from this.

    Returns a tuple of None, None if the DCASE class label cannot be inferred.

    Arguments:
        filepath: the path to infer the class label from

    Examples:
        >>> fpath = "/AudibleLight/resources/soundevents/maleSpeech/train/Male_speech_and_man_speaking/67669.wav"
        >>> cls, idx = infer_dcase_label_idx_from_filepath(fpath)
        >>> print(cls, idx)
        tuple("maleSpeech", 1)
    """
    # Coerce paths
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    cls, idx = None, None
    # Search for the label key in the path
    for part in filepath.parts:
        if part in DCASE_SOUND_EVENT_CLASSES.keys():

            # Update the variables, only if we haven't already done so
            if not cls and not idx:
                cls = part
                idx = DCASE_SOUND_EVENT_CLASSES[cls]

            # Raise if we have multiple possible matches
            else:
                raise ValueError(
                    f"Found multiple possible DCASE classes for filepath {str(filepath)}. "
                    f"This filepath matches both classes {cls} and {part}. "
                    f"Please adjust your filepaths as necessary so that they only contain one DCASE class."
                )

    # This will just return None, None if no matches
    return idx, cls


def _infer_missing_dcase_values(
    class_id: Optional[int], class_label: Optional[str]
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
        emitters: Optional[Union[list[Emitter], Emitter, list[dict]]] = None,
        augmentations: Optional[
            Union[Iterable[Type[EventAugmentation]], Type[EventAugmentation]]
        ] = None,
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

        Arguments:
            filepath: Path to the audio file.
            alias: Label to refer to this Event by inside the Scene
            emitters: List of Emitter objects associated with this Event.
                If not provided, `register_emitters` must be called prior to rendering any Scene this Event is in.
            augmentations: Iterable of EventAugmentation objects associated with this Event.
                If not provided, EventAugmentations can be registered later by calling `register_augmentations`.
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
        self.filepath = utils.sanitise_filepath(
            filepath
        )  # will raise an error if not found on disk, coerces to Path
        self.audio = None  # will be loaded when calling `load_audio` for the first time
        self.snr = snr
        self.sample_rate = utils.sanitise_positive_number(sample_rate)
        self.alias = alias

        # Valid augmentations will be stored here and called when loading audio
        self.augmentations = []
        if augmentations is not None:
            self.register_augmentations(augmentations)

        # Spatial audio attributes, set in the synthesizer
        self.spatial_audio = None

        # Spatial attributes
        self.spatial_resolution = spatial_resolution
        self.spatial_velocity = spatial_velocity

        # Metadata attributes
        self.filename = self.filepath.name

        #  Attempt to infer class ID and labels in cases where only one is provided
        if class_id or class_label:
            self.class_id, self.class_label = _infer_missing_dcase_values(
                class_id, class_label
            )
        # Otherwise, try and infer both the ID and the label from the filepath
        else:
            self.class_id, self.class_label = infer_dcase_label_idx_from_filepath(
                self.filepath
            )

        # Get the full duration of the audio file
        self.audio_full_duration = utils.sanitise_positive_number(
            librosa.get_duration(path=self.filepath)
        )
        # Event start is the offset from the start of the audio file
        self.event_start = self._parse_audio_start(event_start)
        # Scene start is the time the event starts in the scene
        self.scene_start = (
            utils.sanitise_positive_number(scene_start)
            if scene_start is not None
            else 0.0
        )
        # Safely parse the duration of the audio file with an optional override
        self.duration = self._parse_duration(duration)

        # Now we can safely get the ending time of the event
        self.event_end = self.event_start + self.duration
        self.scene_end = self.scene_start + self.duration

        # Emitters: these will be set when calling `register_emitters`
        self.emitters = None
        self.is_moving = None

        # Coordinate metadata: these will be set when calling `register_emitters`
        self.start_coordinates_absolute = None
        self.end_coordinates_absolute = None
        self.start_coordinates_relative_cartesian = None
        self.end_coordinates_relative_cartesian = None
        self.start_coordinates_relative_polar = None
        self.end_coordinates_relative_polar = None

        # If we've provided emitters, go ahead and register them now
        #  Otherwise, we must provide these before calling any synthesis functions
        if emitters is not None:
            self.register_emitters(emitters)
        #  This is mostly for backwards compatibility when `emitters` was a required argument of Event.__init__
        #  In practice, it allows us to get the duration of an event (etc.) before adding emitters
        #  This can be useful in cases where we need the duration of an event before creating emitters
        #  such as when defining the trajectory of a moving event

    def register_augmentations(
        self,
        augmentations: Union[
            Iterable[Type[EventAugmentation]], Type[EventAugmentation]
        ],
    ) -> None:
        """
        Register augmentations associated with this Event.

        Arguments:
            augmentations: Iterable of augmentations to register, or a single EventAugmentation type

        Returns:
            None
        """
        # Handle single augmentations
        if not isinstance(augmentations, (list, tuple, set)):
            augmentations = [augmentations]

        # Iterate over all augmentations
        for aug in augmentations:

            # If the augmentation hasn't been initialised yet, try doing this now
            if isinstance(aug, type):
                try:
                    aug = aug(sample_rate=self.sample_rate)
                except Exception as e:
                    raise e

            # Check that the sample rate is valid
            if aug.sample_rate != self.sample_rate:
                raise ValueError(
                    f"Augmentation has mismatching sample rate! "
                    f"expected {self.sample_rate}, got {aug.sample_rate}"
                )

            # Validate the augmentation and add it in if it's OK
            validate_event_augmentation(aug)
            self.augmentations.append(aug)

        # Whenever we register augmentations, we should also invalidate any cached audio
        #  This will force us to reload the audio and apply the augmentations again
        #  when we call `self.load_audio`.
        self.audio = None
        self.spatial_audio = None

    def register_emitters(
        self,
        emitters: Union[list[Emitter], Emitter, list[dict]],
    ) -> None:
        """
        Registers emitters associated with this event.

        Arguments:
            emitters: List of Emitter objects associated with this event.

        Returns:
            None
        """

        # Register the emitter objects to the current class
        self.emitters = self._parse_emitters(emitters)

        #  If more than one emitter, the sound source is moving; if only one emitter, the sound source is stationary
        self.is_moving = len(self.emitters) > 1

        # We presume that the list of emitter objects is "sorted"
        #  i.e., that the first emitter corresponds to the start position and the last to the end
        first_emitter = self.emitters[0]
        self.start_coordinates_absolute = first_emitter.coordinates_absolute
        self.start_coordinates_relative_cartesian = (
            first_emitter.coordinates_relative_cartesian
        )
        self.start_coordinates_relative_polar = first_emitter.coordinates_relative_polar

        # Set the ending coordinates: if the object is not moving, these are the same as the starting coordinates.
        if self.is_moving:
            last_emitter = self.emitters[-1]
            self.end_coordinates_absolute = last_emitter.coordinates_absolute
            self.end_coordinates_relative_cartesian = (
                last_emitter.coordinates_relative_cartesian
            )
            self.end_coordinates_relative_polar = (
                last_emitter.coordinates_relative_polar
            )
        else:
            self.end_coordinates_absolute = self.start_coordinates_absolute
            self.end_coordinates_relative_cartesian = (
                self.start_coordinates_relative_cartesian
            )
            self.end_coordinates_relative_polar = self.start_coordinates_relative_polar

    def __str__(self) -> str:
        """
        Returns a string representation of the scene
        """
        loaded = "loaded" if self.is_audio_loaded else "unloaded"
        moving = "Moving" if self.is_moving else "Static"
        emits = "no " if self.emitters is None else len(self)
        return (
            f"{moving} 'Event' with alias '{self.alias}',"
            f" audio file '{self.filepath}' ({loaded}, {len(self.augmentations)} augmentations), {emits} emitter(s)."
        )

    def __repr__(self) -> str:
        """
        Returns representation of the scene as a JSON
        """
        return utils.repr_as_json(self)

    def __eq__(self, other: Any) -> bool:
        """
        Compare two Event objects for equality.

        Returns:
            bool: True if two Event objects are equal, False otherwise
        """

        # Non-Event objects are always not equal
        if not isinstance(other, Event):
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
            exclude_paths="emitters",
            ignore_numeric_type_changes=True,
        )

        # If there is no difference, there should be no keys in the deepdiff object
        return len(diff) == 0

    def __len__(self) -> int:
        """
        Get the number of Emitters registered to this Event, alias for `len(Event.emitters)`.
        """
        if self.has_emitters:
            return len(self.emitters)
        else:
            raise ValueError(
                "Cannot get length of an Event object without registered emitters."
            )

    @property
    def has_emitters(self) -> bool:
        """
        Returns True if Event has valid emitters associated with it, False otherwise
        """
        return self.emitters is not None and all(
            isinstance(e, Emitter) for e in self.emitters
        )

    @property
    def is_audio_loaded(self) -> bool:
        """
        Returns True if audio is loaded and valid (see `librosa.util.valid_audio` for more detail).
        """
        return self.audio is not None and librosa.util.valid_audio(self.audio)

    # noinspection PyUnreachableCode
    def _parse_emitters(
        self,
        emitters: Union[
            Emitter, list[Emitter], list[dict], list[list], list[np.ndarray]
        ],
    ) -> list[Emitter]:
        """
        Safely handle coercing objects to a list of `Emitter`s
        """
        # List of emitter objects associated with this event
        if isinstance(emitters, Emitter):
            return [emitters]  # pad to a list

        elif isinstance(emitters, list):
            if len(emitters) < 1:
                raise ValueError("At least one emitter must be provided")

            # Parse list of dictionaries
            elif all(isinstance(em, dict) for em in emitters):
                emitters: list[dict]
                return [Emitter.from_dict(dic) for dic in emitters]

            # Parse list of emitters
            elif all(isinstance(em, Emitter) for em in emitters):
                return emitters

            # Parse list of arrays or lists by creating new emitter objects with the same values
            elif all(isinstance(em, (np.ndarray, list)) for em in emitters):
                return [
                    Emitter(
                        alias=self.alias,
                        coordinates_absolute=utils.sanitise_coordinates(em),
                    )
                    for em in emitters
                ]

            else:
                raise TypeError(
                    "Cannot parse emitter with type {}".format(type(emitters[0]))
                )

        else:
            raise TypeError("Cannot parse emitters with type {}".format(type(emitters)))

    def _parse_audio_start(self, audio_start: Optional[utils.Numeric] = None) -> float:
        """
        Safely handle getting the start/offset time for an audio event, with an optional override.
        """
        if audio_start is None:
            event_start_ = 0.0
        # Raise a warning and revert to 0 seconds when passed start time exceeds total duration of the audio file
        elif audio_start > self.audio_full_duration:
            logger.warning(
                f"Event start time ({audio_start:.2f} seconds) exceeds duration of the audio file "
                f"({self.audio_full_duration:.2f} seconds). Start time will be set to 0."
            )
            event_start_ = 0.0
        else:
            event_start_ = audio_start
        return utils.sanitise_positive_number(event_start_)

    def _parse_duration(self, duration: Optional[float] = None) -> float:
        """
        Safely handle getting the duration of an audio file, with an optional override.
        """
        # If we haven't passed in an override, just use the full duration of the audio, minus the offset
        #  Set a cap so that long audio is truncated e.g. to 10 seconds
        if duration is None:
            # TODO: think about using a constant value here
            #  So, e.g., music sound events get truncated to 10 seconds
            return utils.sanitise_positive_number(
                self.audio_full_duration - self.event_start
            )
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
    def load_audio(self, ignore_cache: Optional[bool] = False) -> np.ndarray:
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
            audio_raw, _ = librosa.load(
                self.filepath,
                sr=self.sample_rate,
                mono=True,
                offset=self.event_start,
                duration=self.duration,
                dtype=np.float32,
            )

        # Apply all augmentations to the audio
        audio_out = audio_raw.copy()
        for aug in self.augmentations:
            audio_out = aug(audio_out)

        self.audio = audio_out
        return self.audio

    def to_dict(self) -> dict:
        """
        Returns metadata for this Event as a dictionary.
        """

        def coerce(inp):
            if isinstance(inp, dict):
                return {k_: coerce(v_) for k_, v_ in inp.items()} if inp else None
            elif isinstance(inp, np.ndarray):
                return inp.tolist()
            else:
                return inp

        if not self.has_emitters:
            raise ValueError("Cannot dump metadata for an Event with no Emitters!")

        # Get relative positions from emitters
        relative_positions = {}
        for emitter in self.emitters:
            for k, v in emitter.coordinates_relative_polar.items():
                if k in relative_positions.keys():
                    relative_positions[k].append(coerce(v)[0])
                else:
                    relative_positions[k] = [coerce(v)[0]]

        return dict(
            # Metadata
            alias=self.alias,
            filename=str(self.filename),
            filepath=str(self.filepath),
            class_id=self.class_id,
            class_label=self.class_label,
            is_moving=self.is_moving,
            # Audio stuff
            scene_start=self.scene_start,
            scene_end=self.scene_end,
            event_start=self.event_start,
            event_end=self.event_end,
            duration=self.duration,
            snr=self.snr,
            sample_rate=self.sample_rate,
            # Spatial stuff (inherited from Emitter objects)
            spatial_resolution=self.spatial_resolution if self.is_moving else None,
            spatial_velocity=self.spatial_velocity if self.is_moving else None,
            # start_coordinates=coerce(self.start_coordinates_absolute),
            # end_coordinates=coerce(self.end_coordinates_absolute),
            num_emitters=len(self.emitters),
            # Include the actual emitters as well, to enable unserialisation
            emitters=[coerce(v.coordinates_absolute) for v in self.emitters],
            emitters_relative=relative_positions,
            # Include the augmentation objects
            augmentations=[aug.to_dict() for aug in self.augmentations],
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate a `Event` from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `Event`.

        Returns:
            Event instance.
        """

        # Sanitise inputs
        for k in [
            "alias",
            "filepath",
            "emitters",
            "snr",
            "duration",
            "event_start",
            "scene_start",
            "scene_end",
        ]:
            if k not in input_dict:
                raise KeyError(f"Missing key: '{k}'")

        # Define emitters list from relative + absolute positions
        #  This is pretty horrible, but I can't think of another way to do it
        emitters_list = []
        emitters_relative = input_dict["emitters_relative"]
        for emitter_idx, emitter in enumerate(input_dict["emitters"]):
            obj = Emitter(alias=input_dict["alias"], coordinates_absolute=emitter)
            obj.coordinates_relative_polar = OrderedDict(
                {
                    k: np.array([emitters_relative[k][emitter_idx]])
                    for k in emitters_relative.keys()
                }
            )
            obj.coordinates_relative_cartesian = OrderedDict(
                {
                    k: utils.polar_to_cartesian(emitters_relative[k][emitter_idx])
                    for k in emitters_relative.keys()
                }
            )
            emitters_list.append(obj)

        # Reconstruct augmentations
        augs = []
        if "augmentations" in input_dict.keys():
            for aug in input_dict["augmentations"]:
                augs.append(EventAugmentation.from_dict(aug))

        # Instantiate the event and return
        return cls(
            alias=input_dict["alias"],
            filepath=input_dict["filepath"],
            emitters=emitters_list,
            augmentations=augs,
            scene_start=input_dict["scene_start"],
            event_start=input_dict["event_start"],
            duration=input_dict["duration"],
            snr=input_dict["snr"],
            sample_rate=input_dict["sample_rate"],
            class_id=input_dict["class_id"],
            class_label=input_dict["class_label"],
            spatial_resolution=input_dict["spatial_resolution"],
            spatial_velocity=input_dict["spatial_velocity"],
        )

    def get_augmentation(self, idx: int) -> Type[EventAugmentation]:
        """
        Gets a single augmentation associated with this Event by its integer index.
        """
        try:
            return self.augmentations[idx]
        except IndexError:
            raise IndexError("No augmentation with index {}".format(idx))

    def get_augmentations(self) -> list[Type[EventAugmentation]]:
        """
        Gets all augmentations associated with this Event.
        """
        return self.augmentations

    def get_emitter(self, idx: int) -> Emitter:
        """
        Gets a single Emitter associated with this Event by its integer index
        """
        try:
            return self.emitters[idx]
        except (IndexError, TypeError):
            raise IndexError("No emitter with index {}".format(idx))

    def get_emitters(self) -> list[Emitter]:
        """
        Gets all emitters associated with this Event.
        """
        return self.emitters if self.emitters is not None else []

    def clear_augmentation(self, idx: int) -> None:
        """
        Clears an augmentation at a given index
        """
        try:
            del self.augmentations[idx]
        except IndexError:
            raise IndexError("No augmentation found at index {idx}".format(idx=idx))
        else:
            # Invalidate any cached audio
            self.audio = None
            self.spatial_audio = None

    def clear_augmentations(self) -> None:
        """
        Removes all augmentations associated with this Event.
        """
        if len(self.augmentations) > 0:
            self.augmentations = []
            # Invalidate any cached audio
            self.audio = None
            self.spatial_audio = None
