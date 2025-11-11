#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core modules and functions for generation and synthesis."""

import json
import random
from collections import OrderedDict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Type, Union

import numpy as np
import soundfile as sf
from deepdiff import DeepDiff
from loguru import logger
from scipy import stats

from audiblelight import config, custom_types, utils
from audiblelight.ambience import Ambience
from audiblelight.augmentation import ALL_EVENT_AUGMENTATIONS, EventAugmentation
from audiblelight.class_mappings import (
    ClassMapping,
    TClassMapping,
    sanitize_class_mapping,
)
from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import Emitter, WorldState, get_worldstate_from_string


class Scene:
    """
    Initializes a Scene.

    The Scene object is the highest level object within AudibleLight. It holds information relating to the current
    WorldState (including a 3D mesh, alongside listeners and sound sources) and any sound Event objects within it.
    """

    def __init__(
        self,
        duration: custom_types.Numeric,
        backend: Union[str, "WorldState"],
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        fg_path: Optional[Union[str, Path]] = None,
        bg_path: Optional[Union[str, Path]] = None,
        allow_duplicate_audios: bool = True,
        allow_same_class_events: bool = True,
        ref_db: Optional[custom_types.Numeric] = config.DEFAULT_REF_DB,
        scene_start_dist: Optional[custom_types.DistributionLike] = None,
        event_start_dist: Optional[custom_types.DistributionLike] = None,
        event_duration_dist: Optional[custom_types.DistributionLike] = None,
        event_velocity_dist: Optional[custom_types.DistributionLike] = None,
        event_resolution_dist: Optional[custom_types.DistributionLike] = None,
        snr_dist: Optional[custom_types.DistributionLike] = None,
        max_overlap: Optional[custom_types.Numeric] = config.MAX_OVERLAP,
        event_augmentations: Optional[
            Union[
                Iterable[Type[EventAugmentation]],
                Iterable[tuple[Type[EventAugmentation], dict]],
                Type[EventAugmentation],
            ]
        ] = None,
        backend_kwargs: Optional[dict] = None,
        class_mapping: Optional[Union[TClassMapping, dict, str]] = "DCASE2023Task3",
    ):
        """
        Initializes the Scene with a given duration and mesh.

        Arguments:
            duration: the length of time the scene audio should last for.
            backend: the name of the backend to use. Either 'rlr', 'sofa', 'or 'shoebox' are supported.
            fg_path: a directory (or list of directories) pointing to foreground audio. Note that directories will be
                introspected recursively, such that audio files within any subdirectories will be detected also.
            bg_path: a directory (or list of directories pointing to background audio. Note that directories will be
                introspected recursively, such that audio files within any subdirectories will be detected also.
            allow_duplicate_audios: if True (default), the same audio file can appear multiple times in the Scene.
            allow_same_class_events: if True (default), multiple Events from the same class may be added to the Scene.
            ref_db: reference decibel level for scene noise floor, defaults to -65 dB
            scene_start_dist: distribution-like object or callable used to sample starting times for any Event objects
                applied to the scene. If not provided, will be a uniform distribution between 0 and `duration`
            event_start_dist: distribution-like object used to sample starting (offset) times for Event audio files.
                If not provided, Event audio files will always start at 0 seconds. Note that this can be overridden
                by passing a value into `Scene.add_event(event_start=...)`
            event_duration_dist: distribution-like object used to sample Event audio duration times. If not provided,
                Event audio files will always use their full duration. Note that this can be overridden by passing a
                value into `Scene.add_event(duration=...)`
            event_velocity_dist: distribution-like object used to sample Event spatial velocities. If not provided, a
                uniform distribution between 0.5 and 2.0 metres-per-second will be used.
            event_resolution_dist: distribution-like object used to sample Event spatial resolutions. If not provided,
                a uniform distribution between 1.0 and 4.0 Hz (i.e., IRs-per-second) will be used.
            snr_dist: distribution-like object used to sample Event signal-to-noise ratios. If not provided, a uniform
                distribution between 5 and 30 will be used.
            max_overlap: the maximum number of overlapping audio Events allowed in the Scene, defaults to 2.
            event_augmentations: an iterable of `audiblelight.EventAugmentation` objects that can be applied to Event
                objects. The number of augmentations sampled from this list can be controlled by setting the value of
                `augmentations` when calling `Scene.add_event`, i.e. `Scene.add_event(augmentations=3)` will sample
                3 random augmentations from `event_augmentations` and apply them to the Event.
            backend_kwargs: keyword arguments passed to `audiblelight.WorldState`.
            class_mapping: a mapping used to map class names to indices, and vice versa. Can be a subclass of
                `audiblelight.class_mapping.ClassMapping`, `dict`, or `str`. Defaults to DCASE 2023, task 3 mapping
        """

        # Set attributes passed in by the user
        self.duration = utils.sanitise_positive_number(duration)
        # Raise a warning when the duration is very short.
        if self.duration < config.WARN_WHEN_SCENE_DURATION_BELOW:
            logger.warning(
                f"The duration for this Scene is very short ({duration:.2f} seconds). "
                f"You may encounter issues with Events overlapping or being truncated to fit the "
                f"duration of the Scene. It is recommended to increase the duration to at least "
                f"{config.WARN_WHEN_SCENE_DURATION_BELOW} seconds."
            )
        self.ref_db = self._sanitise_ref_db(ref_db)
        # Time overlaps (we could include a space overlaps parameter too)
        self.max_overlap = utils.sanitise_positive_number(max_overlap, cast_to=int)

        # Set sample rate correctly
        self.sample_rate = utils.sanitise_positive_number(sample_rate, cast_to=int)

        # Instantiate the `WorldState` object, which loads the mesh and sets up the ray-tracing engine
        if backend_kwargs is None:
            backend_kwargs = {}

        # Coercing backend from string
        if isinstance(backend, str):
            desired_state = get_worldstate_from_string(backend)
            utils.validate_kwargs(desired_state.__init__, **backend_kwargs)
            self.state = desired_state(sample_rate=self.sample_rate, **backend_kwargs)

        # Otherwise, using backend directly
        elif issubclass(type(backend), WorldState):
            be_sr = getattr(backend, "sample_rate", None)
            if not be_sr or be_sr != self.sample_rate:
                raise ValueError(
                    f"Mismatching backend sample rate: expected {self.sample_rate}, got {be_sr}"
                )
            self.state = backend

        else:
            raise TypeError(
                f"Expected 'backend' to be a string or an *instance* of WorldState subclass, "
                f"but got {type(backend)} instead."
            )

        # Grab some attributes from the WorldState to make them easier to access
        self.mesh = self.state.mesh
        # self.irs = self.state.irs

        # Define defaults for all distributions
        #  Events can start any time within the duration of the scene, minus some padding
        if scene_start_dist is None:
            scene_start_dist = stats.uniform(0.0, self.duration - 1)
        if event_velocity_dist is None:
            event_velocity_dist = stats.uniform(
                config.MIN_EVENT_VELOCITY,
                config.MAX_EVENT_VELOCITY - config.MIN_EVENT_VELOCITY,
            )
        if event_resolution_dist is None:
            event_resolution_dist = stats.uniform(
                config.MIN_EVENT_RESOLUTION,
                config.MAX_EVENT_RESOLUTION - config.MIN_EVENT_RESOLUTION,
            )
        if snr_dist is None:
            snr_dist = stats.uniform(
                config.MIN_EVENT_SNR, config.MAX_EVENT_SNR - config.MIN_EVENT_SNR
            )

        # No distribution for `event_start` and `event_distribution`
        #  Unless a distribution is passed, we default to using the full duration of the audio (capped at 10 seconds)
        #  and starting the audio at 0.0 seconds

        # Distributions: these function sanitise the distributions so that they are either `None` or an object
        #  with the `rvs` method. When called, the `rvs` method will return a random variate sampled from the
        #  probability distribution.
        self.scene_start_dist = utils.sanitise_distribution(scene_start_dist)
        self.event_start_dist = utils.sanitise_distribution(event_start_dist)
        self.event_duration_dist = utils.sanitise_distribution(event_duration_dist)
        self.event_velocity_dist = utils.sanitise_distribution(event_velocity_dist)
        self.event_resolution_dist = utils.sanitise_distribution(event_resolution_dist)
        self.snr_dist = utils.sanitise_distribution(snr_dist)

        # Parse foreground audio directory (or directories) and obtain all valid audio files from within it
        self.fg_paths = (
            self._parse_audio_directories(fg_path) if fg_path is not None else []
        )
        self.fg_audios = self._introspect_audio_directories(self.fg_paths)
        # Do the same for background audio
        self.bg_paths = (
            self._parse_audio_directories(bg_path) if bg_path is not None else []
        )
        self.bg_audios = self._introspect_audio_directories(self.bg_paths)

        # If False, we'll ensure that all randomly sampled event/ambience audio is unique when sampling
        self.allow_duplicate_audios = allow_duplicate_audios

        # If False, we'll ensure that all randomly sampled event audio is from a unique class
        self.allow_same_class_events = allow_same_class_events

        # Events will be stored within here
        self.events = OrderedDict()

        # Event augmentations
        self.event_augmentations = []
        if event_augmentations is not None:
            self.event_augmentations = self._parse_event_augmentations(
                event_augmentations
            )

        # Background noise
        #  if not None (i.e., with a call to `add_ambience`), will be added to audio when synthesising
        self.ambience = OrderedDict()

        # Spatialized audio
        #  Note that this is a dictionary to support multiple microphones
        self.audio = OrderedDict()

        # Parse class mapping
        self.class_mapping = sanitize_class_mapping(class_mapping)

    @staticmethod
    def _sanitise_ref_db(ref_db: Any) -> int:
        """
        Validate noise floor, in dB, and raise warnings when non-negative.
        """
        if not isinstance(ref_db, custom_types.Numeric):
            raise TypeError(f"Expected `ref_db` to be numeric, but got {type(ref_db)}")
        elif ref_db > 0:
            logger.error(
                f"Provided noise floor is positive; expect clipping to occur (ref_db={ref_db:.2f})"
            )
        return int(ref_db)

    @staticmethod
    def _parse_audio_directories(
        audio_dir: Union[str, Path, list[str], list[Path]]
    ) -> tuple[list[Path], list[Path]]:
        """
        Validate audio directory (or list of directories) and return as a list of Path objects
        """
        if not isinstance(audio_dir, list):
            audio_dir = [audio_dir]
        return utils.sanitise_directories(audio_dir)

    @staticmethod
    def _introspect_audio_directories(audio_dir: list[Path]) -> list[Path]:
        """
        Introspect a list of audio directories to obtain all valid audio files
        """
        audio_paths = []
        for ext in custom_types.AUDIO_EXTS:
            for fg in audio_dir:
                audio_paths.extend((fg.rglob(f"*.{ext}")))
        return utils.sanitise_filepaths(audio_paths)

    def _parse_event_augmentations(
        self,
        event_augmentations: Union[
            Iterable[Type[EventAugmentation]],
            Iterable[tuple[Type[EventAugmentation], dict]],
            Type[EventAugmentation],
        ],
    ) -> list[tuple[Type[EventAugmentation], dict]]:
        """
        Parse user-provided event augmentations.

        The result is a list of tuples with the form (EventAugmentationType, dict), where the dict contains
        pre-validated keyword arguments that are passed to the EventAugmentation every time it is constructed.

        This enables the user (for instance) to override the default values or distributions used within an
        EventAugmentation class.

        Arguments:
            event_augmentations: the augmentation custom_types. Can be a list of types, a list of tuples, or a single type.

        Returns:
            list[tuple]: the list of augmentation types and validated kwargs
        """
        # Coerce single types to a list
        if not isinstance(event_augmentations, (tuple, list, np.ndarray)):
            event_augmentations = [event_augmentations]

        sanitised_event_augmentations = []

        for maybe_iter in event_augmentations:
            # We've passed in tuples of type (AugmentationClass, augmentation_kwargs)
            if (
                isinstance(maybe_iter, (tuple, list, np.ndarray))
                and len(maybe_iter) == 2
            ):
                aug_type, kwargs_dict = maybe_iter

            # We've just passed in the AugmentationClass, so use default kwargs for it
            elif isinstance(maybe_iter, type):
                aug_type = maybe_iter
                kwargs_dict = dict()

            # We don't know what's been passed in
            else:
                raise TypeError(
                    f"Expected a tuple or EventAugmentation type but got {type(maybe_iter)}"
                )

            # Validate the augmentation is a subclass of the parent object
            if not issubclass(aug_type, EventAugmentation):
                raise TypeError(
                    f"Expected an EventAugmentation subclass but got {type(aug_type)}"
                )

            # Ensure sample rate for the augmentation is set correctly
            if (
                "sample_rate" in kwargs_dict
                and kwargs_dict["sample_rate"] != self.sample_rate
            ):
                raise ValueError(
                    f"Expected a sample rate {self.sample_rate}, but got {kwargs_dict['sample_rate']}"
                )
            kwargs_dict["sample_rate"] = self.sample_rate

            # Validate the kwargs for the augmentation and store
            utils.validate_kwargs(aug_type, **kwargs_dict)
            sanitised_event_augmentations.append((aug_type, kwargs_dict))

        # Do not drop duplicates
        return sanitised_event_augmentations

    def __eq__(self, other: Any) -> bool:
        """
        Compare two Scene objects for equality.

        Internally, we convert both objects to a dictionary, and then use the `deepdiff` package to compare them, with
        some additional logic to account e.g. for significant digits and values that will always be different (e.g.,
        creation time).

        Arguments:
            other: the object to compare the current `Scene` object against

        Returns:
            bool: True if the Scene objects are equivalent, False otherwise
        """

        # Non-Scene objects are always not equal
        if not isinstance(other, Scene):
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
            exclude_paths="creation_time",
            ignore_numeric_type_changes=True,
        )

        # If there is no difference, there should be no keys in the deepdiff object
        return len(diff) == 0

    def __len__(self) -> int:
        """
        Returns the number of events in the scene
        """
        return len(self.events)

    def __str__(self) -> str:
        """
        Returns a string representation of the scene
        """
        return (
            f"'Scene' with mesh '{self.state.mesh.metadata['fpath']}': "
            f"{len(self)} events, {len(self.state.microphones)} microphones, {self.state.num_emitters} emitters."
        )

    def __repr__(self) -> str:
        """
        Returns representation of the scene as a JSON
        """
        return utils.repr_as_json(self)

    def __getitem__(self, alias_or_idx: Union[str, int]) -> Event:
        """
        An alternative for `self.get_event(alias) or `self.events[alias]`
        """
        return self.get_event(alias_or_idx)

    def __iter__(self) -> Iterator[Event]:
        """
        Yields an iterator of Event objects from the current scene

        Examples:
            >>> test_scene = Scene(...)
            >>> for n in range(9):
            >>>     test_scene.add_event_static(...)
            >>> for ev in test_scene:
            >>>     assert isinstance(ev, Event)
        """
        yield from self.get_events()

    def add_microphone(self, **kwargs) -> None:
        """
        Add a microphone to the WorldState.

        An alias for `WorldState.add_microphone`: see that method for a full description.
        """
        utils.validate_kwargs(self.state.add_microphone, **kwargs)
        self.state.add_microphone(**kwargs)

    def add_microphones(self, **kwargs) -> None:
        """
        Add microphones to the WorldState.

        An alias for `WorldState.add_microphones`: see that method for a full description.
        """
        utils.validate_kwargs(self.state.add_microphones, **kwargs)
        self.state.add_microphones(**kwargs)

    def add_microphone_and_emitter(self, **kwargs) -> None:
        """
        Add both a microphone and emitter with specified relationship.

        An alias for `WorldState.add_microphone_and_emitter`: see that method for a full description.
        """
        utils.validate_kwargs(self.state.add_microphone_and_emitter, **kwargs)
        self.state.add_microphone_and_emitter(**kwargs)

    def add_emitter(self, **kwargs):
        """
        Add an emitter to the WorldState.

        An alias for `WorldState.add_emitter`: see that method for a full description.
        """
        logger.warning(
            "Adding an Emitter directly to the WorldState is not recommended. Instead, use "
            "`Scene.add_event`, which will create an Event and add any required Emitters to the WorldState."
        )
        utils.validate_kwargs(self.state.add_emitter, **kwargs)
        self.state.add_emitter(**kwargs)

    def add_emitters(self, **kwargs):
        """
        Add emitters to the WorldState.

        An alias for `WorldState.add_emitters`: see that method for a full description.
        """
        logger.warning(
            "Adding Emitters directly to the WorldState is not recommended. Instead, use "
            "`Scene.add_event`, which will create Events and add any required Emitters to the WorldState."
        )
        utils.validate_kwargs(self.state.add_emitters, **kwargs)
        self.state.add_emitters(**kwargs)

    def add_ambience(
        self,
        filepath: Optional[Union[str, Path]] = None,
        noise: Optional[Union[str, custom_types.Numeric]] = None,
        channels: Optional[int] = None,
        ref_db: Optional[custom_types.Numeric] = None,
        alias: Optional[str] = None,
        **kwargs,
    ):
        """
        Add ambient noise to the WorldState.

        The ambience can be either a file on the disk (in which case filepath must not be None) or a type of noise
        "color" such as white, red, or blue (in which case noise must not be None). The number of channels can be
        provided directly or will be inferred from the microphones added to the state, when this is possible.

        Arguments:
            channels (int): the number of channels to generate noise for. If None, will be inferred from available mics.
            filepath (str or Path): a path to an audio file on the disk. If None (and `noise` is None), will try and
                sample a random audio file from `Scene.bg_audios`.
            noise (str): either the type of noise to generate, e.g. "white", "red", or an arbitrary numeric exponent to
                use when generating noise with `powerlaw_psd_gaussian`.
            ref_db (Numeric): the noise floor, in decibels
            alias (str): string reference to refer to this `Ambience` object inside `Scene.ambience`
            kwargs: additional keyword arguments passed to `audiblelight.ambience.powerlaw_psd_gaussian`
        """
        # If the number of channels is not provided, try and get this from the number of microphone capsules
        if channels is None:
            if len(self.state.microphones) == 0:
                raise ValueError(
                    "Cannot infer Ambience channels when no microphones have been added to the WorldState."
                )

            # Ray-tracing contexts: mic can have arbitrary channel count
            if self.state.name.upper() == "RLR":
                available_mics = [
                    mic.n_capsules for mic in self.state.microphones.values()
                ]
                # Raise an error when added microphones have a different number of channels
                if not all([a == available_mics[0] for a in available_mics]):
                    raise ValueError(
                        "Cannot infer Ambience channels when available microphones have different number of capsules"
                    )
                else:
                    channels = available_mics[0]
            # SOFA contexts: mic currently only has 4 channels
            # TODO: check this: some SOFA files may have more than 4 channels, which will break downstream?
            else:
                channels = 4

        # Get the alias for this ambience event: either default or user-provided
        alias = (
            utils.get_default_alias("ambience", self.ambience)
            if alias is None
            else alias
        )
        if alias in self.ambience:
            raise KeyError(
                f"Ambience with alias '{alias}' has already been added to the Scene!"
            )

        # Handling filepaths
        if noise is None:
            # Get a random filepath when neither noise or filepath provided
            if filepath is None:
                filepath = self._get_random_audio(self.bg_audios)
            # Otherwise, check provided filepath is valid
            else:
                filepath = utils.sanitise_filepath(filepath)

            # If we don't want to allow for duplicate filepaths, check this now
            if not self.allow_duplicate_audios:
                seen_audios = self._get_used_audios()
                if filepath in seen_audios:
                    raise ValueError(
                        f"Audio file {str(filepath.resolve())} has already been added to the Scene. "
                        f"Either increase the number of `bg_paths` in Scene.__init__, "
                        f"choose a different audio file, "
                        f"or set `Scene.allow_duplicate_audios=False`."
                    )

        # Add the ambience to the dictionary
        self.ambience[alias] = Ambience(
            channels=channels,
            duration=self.duration,
            sample_rate=self.sample_rate,
            noise=noise,
            filepath=filepath,
            alias=alias,
            ref_db=ref_db if ref_db is not None else self.ref_db,
            **kwargs,
        )

    # noinspection PyProtectedMember
    def _try_add_event(self, **event_kwargs) -> bool:
        """
        Tries to add an Event with given kwargs.

        Returns:
            bool: True if successful, False otherwise.
        """
        # Grab the alias: this should always be present inside the dictionary
        alias = event_kwargs["alias"]

        # Use only 1 placement attempt if all overrides are present
        has_overrides = all(
            k is not None in event_kwargs
            for k in ("scene_start", "event_start", "duration")
        )
        max_place_attempts = (
            event_kwargs.get("max_place_attempts", config.MAX_PLACE_ATTEMPTS)
            if not has_overrides
            else 1
        )

        # Pre-resolve all user-specified override values (only done once)
        overrides = {
            "filepath": event_kwargs.get("filepath"),
            "scene_start": event_kwargs.get("scene_start"),
            "event_start": event_kwargs.get("event_start"),
            "duration": event_kwargs.get("duration"),
            "snr": event_kwargs.get("snr"),
            "spatial_velocity": event_kwargs.get("spatial_velocity"),
            "spatial_resolution": event_kwargs.get("spatial_resolution"),
        }

        for _ in range(max_place_attempts):
            # Copy once per attempt
            current_kws = event_kwargs.copy()

            # Get a random audio file for this attempt
            if overrides["filepath"] is None:
                current_kws["filepath"] = self._get_random_audio(self.fg_audios)

            # If we haven't passed in a duration override OR a distribution, default to using the full audio duration
            if overrides["duration"] is None and self.event_duration_dist is None:
                current_kws["duration"] = None
            # Otherwise, try and sample from the distribution or use the override
            else:
                current_kws["duration"] = utils.sample_distribution(
                    self.event_duration_dist, overrides["duration"]
                )

            # Do the same for event start time
            if overrides["event_start"] is None and self.event_start_dist is None:
                current_kws["event_start"] = None
            else:
                current_kws["event_start"] = utils.sample_distribution(
                    self.event_start_dist, overrides["event_start"]
                )

            # Sample values (with fallback to override if provided)
            current_kws.update(
                {
                    "scene_start": utils.sample_distribution(
                        self.scene_start_dist, overrides["scene_start"]
                    ),
                    "snr": utils.sample_distribution(self.snr_dist, overrides["snr"]),
                    "spatial_velocity": utils.sample_distribution(
                        self.event_velocity_dist, overrides["spatial_velocity"]
                    ),
                    "spatial_resolution": utils.sample_distribution(
                        self.event_resolution_dist, overrides["spatial_resolution"]
                    ),
                }
            )

            # Create the event with the current keywords
            #  Need to strip out arguments that are only valid for adding emitters to the backend
            valid_event_kwargs = utils.get_valid_kwargs(Event.__init__)
            current_event = Event(
                **{k: v for k, v in current_kws.items() if k in valid_event_kwargs}
            )

            # Reject this attempt if overlap would be exceeded
            if self._would_exceed_temporal_overlap(
                current_event.scene_start, current_event.scene_end
            ):
                continue

            # Reject this attempt if the duration exceeds the duration of the scene
            if current_event.scene_end > self.duration:
                continue

            # Now, we can go ahead and define the emitters for the current run

            # Static event
            if event_kwargs.get("shape") == "static":
                # TODO: **dirty hack**: we hardcode the maximum place attempts to 1, so that we try to add the emitter
                #  once with the current set of parameters only, and skip over to the next set on a ValueError
                emitter_kwargs = dict(
                    position=event_kwargs["position"],
                    alias=alias,
                    mic=event_kwargs["mic"],
                    ensure_direct_path=event_kwargs["ensure_direct_path"],
                    keep_existing=True,
                    max_place_attempts=1,
                )

                # Add the emitters associated with the event to the worldstate
                #  This will perform spatial logic checks for e.g. ensuring that the emitter won't collide with anything
                #  else inside the mesh, such as another emitter, microphone, or the mesh itself
                utils.validate_kwargs(self.state.add_emitter, **emitter_kwargs)
                try:
                    self.state.add_emitter(**emitter_kwargs)
                except ValueError:
                    continue

            # Moving event
            else:
                # Update the kwargs we'll use to create the trajectory with parameters from the event
                # TODO: **dirty hack**: we hardcode the maximum place attempts to 1, so that we try to add the emitter
                #  once with the current set of parameters only, and skip over to the next set on a ValueError
                emitter_kwargs = dict(
                    duration=current_event.duration,
                    velocity=current_event.spatial_velocity,
                    resolution=current_event.spatial_resolution,
                    shape=current_event.shape,
                    starting_position=event_kwargs["starting_position"],
                    ensure_direct_path=event_kwargs["ensure_direct_path"],
                    max_place_attempts=1,
                )
                utils.validate_kwargs(self.state.define_trajectory, **emitter_kwargs)

                # Define the trajectory
                try:
                    trajectory = self.state.define_trajectory(**emitter_kwargs)
                except ValueError:
                    continue

                # Add the emitters to the state with the desired aliases
                #  This just adds the emitters in a loop with no additional checks
                #  We already perform these checks inside `define_trajectory`.
                self.state._add_emitters_without_validating(trajectory, alias)

            # Grab the emitters we just created and register them with the event
            emitters = self.state.get_emitters(alias)
            current_event.register_emitters(emitters)

            # Last thing: store the event
            self.events[alias] = current_event
            return True

        return False

    def _get_used_audios(self) -> list[Path]:
        """
        Gets a list of audio files used in all Ambience and Event objects currently added to the Scene
        """
        # Get all Ambience and Event objects
        events_ambs = self.get_events() + self.get_ambiences()
        # Get audio files: note that `filepath` can be None for ambience objects using noise types instead
        return [ev.filepath for ev in events_ambs if ev.filepath is not None]

    def _get_used_class_ids(self) -> list[int]:
        """
        Gets a list of class IDs used in all Event objects currently added to the scene
        """
        # Just use all Events, Ambience objects do not have class IDs
        return list(set(i.class_id for i in self.get_events()))

    def _get_random_audio(self, audio_paths: Optional[list[Path]] = None) -> Path:
        """
        Gets a path to a random audio file from the provided list of directories

        Arguments:
            audio_paths: a list of audio paths; if not provided, use `fg_audios`
        """
        # Use foreground audio paths by default
        if audio_paths is None:
            audio_paths = self.fg_audios

        # Sanitise all audio paths (converting to pathlib.Path objects, checking they exist...)
        audio_paths = utils.sanitise_filepaths(audio_paths)

        # If we want to ensure that the same audio file is not used multiple times across Events/Ambiences, do this now
        if not self.allow_duplicate_audios:
            seen_audios = self._get_used_audios()
            audio_paths = [i for i in audio_paths if i not in seen_audios]

        # If we want to ensure that a single class cannot appear more than once
        if not self.allow_same_class_events:
            used_class_ids = self._get_used_class_ids()
            audio_paths = [
                ap
                for ap in audio_paths
                if self.class_mapping.infer_label_idx_from_filepath(ap)[0]
                not in used_class_ids
            ]

        # Raise an error when no audio files available
        if len(audio_paths) == 0:
            raise FileNotFoundError(
                "No audio files found to sample from! "
                "Make sure you pass a value to `fg_path` in Scene.__init__`. "
                "You can also try setting `allow_duplicate_audios=True`, or "
                "setting `allow_same_class_events=True`."
            )

        # Choose a random filepath: no need to sanitise, we did this already
        return random.choice(audio_paths)

    def _coerce_polar_position(
        self,
        position: Optional[Union[list, np.ndarray]] = None,
        mic: Optional[str] = None,
    ):
        """
        Coerces a polar position in form [azimuth, elevation, radius] to absolute Cartesian coordinates
        """
        # If we haven't passed a microphone alias
        if mic is None:
            # In cases where we only have one microphone, just use this
            if len(self.state.microphones) == 1:
                mic = list(self.state.microphones.keys())[0]
            elif len(self.state.microphones) == 0:
                raise ValueError(
                    "Cannot set `polar=True` when adding an Event when no microphone has been added to the Scene"
                )
            else:
                raise ValueError(
                    "Must pass a microphone alias when `polar` is True and more than one microphone "
                    "has been added to the Scene"
                )

        # If we haven't passed a position
        if position is None:
            raise ValueError("Must pass a position when `polar` is True")

        # Grab the center position of the mic and add the offset
        return (
            self.state.get_microphone(mic).coordinates_center
            + utils.polar_to_cartesian(position)
        )[0]

    def _get_n_random_event_augmentations(
        self, n_augmentations: custom_types.Numeric
    ) -> list[Type[EventAugmentation]]:
        """
        Given a number N, get N random, unique Event augmentations

        Event augmentations are taken either from `Scene.event_augmentations` or, if this was not set when calling
        `Scene.__init__`, from a master list of valid augmentations inside `audiblelight.augmentations`.

        The returned list of augmentations is guaranteed to be unique.
        """
        # Either use the user provided list of augmentations or the full list
        sample_augs = (
            self.event_augmentations
            if len(self.event_augmentations) > 0
            else [(cls, dict()) for cls in ALL_EVENT_AUGMENTATIONS]
        )

        # Validate the number of augmentations we want
        n_augmentations = utils.sanitise_positive_number(n_augmentations, cast_to=int)

        # Log case where we try to sample more augs than we have available
        if n_augmentations > len(sample_augs):
            logger.warning(
                f"Tried to sample {n_augmentations} random augmentations, but `Scene.event_augmentations` "
                f"only contains {len(sample_augs)} augmentations. Sampling {len(sample_augs)} instead."
            )
            n_augmentations = len(sample_augs)

        # Make a random sample of N augs
        #  This is a list of tuples where the first element is the aug type, and the second is a dictionary of kwargs
        sampled_augs_and_kwargs = random.sample(
            sample_augs,
            k=n_augmentations,
        )

        # Initialise all the augmentations using the provided kwargs
        return [cls(**kws) for cls, kws in sampled_augs_and_kwargs]

    def add_event(
        self,
        event_type: Optional[str] = "static",
        filepath: Optional[Union[str, Path]] = None,
        alias: Optional[str] = None,
        augmentations: Optional[
            Union[
                Iterable[Type[EventAugmentation]],
                Type[EventAugmentation],
                custom_types.Numeric,
            ]
        ] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        trajectory: Optional[np.ndarray] = None,
        mic: Optional[str] = None,
        polar: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        scene_start: Optional[custom_types.Numeric] = None,
        event_start: Optional[custom_types.Numeric] = None,
        duration: Optional[custom_types.Numeric] = None,
        snr: Optional[custom_types.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        shape: Optional[str] = None,
        spatial_resolution: Optional[custom_types.Numeric] = None,
        spatial_velocity: Optional[custom_types.Numeric] = None,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
        **event_kwargs,
    ) -> Event:
        """
        Add an event to the foreground, either "static", "moving", or "predefined".

        Note that the arguments "scene_start", "event_start", "duration", "snr", "spatial_velocity", &
        "spatial_resolution" will (by default) sample from their respective distributions, provided in `Scene.__init__`.
        If a numeric value is provided, this will be treated as an override and used instead of random sampling.

        Arguments:
            event_type (str): the type of event to add, must be either "static", "moving", or "predefined".
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            augmentations: augmentation objects to associate with the Event.
                If a list of EventAugmentation objects or a single EventAugmentation object, these will be passed directly.
                If a number, this many augmentations will be sampled from either `Scene.event_augmentations`, or a master
                list of valid augmentations (defined inside `audiblelight.augmentations`)
                If not provided, EventAugmentations can be registered later by calling `register_augmentations` on the Event.
            position: Location to add the event.
                When `event_type=="static"`, this will be the position of the Event.
                When `event_type=="moving"`, this will be the starting position of the Event.
                When not provided, a random point inside the mesh will be chosen.
            trajectory: The trajectory the moving event will follow, given in Cartesian coordinates inside the mesh.
                Only used when `event_type=="predefined"`. If not provided, will attempt to infer from
                `state.waypoints`.
            mic: String reference to a microphone inside `self.state.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            polar: When True, expects `position` to be provided in [azimuth, elevation, radius] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
            duration: Time the Event audio lasts in seconds. Must be a positive number.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            class_label: Optional label to use for sound event class.
                If not provided, will attempt to infer label from filepath using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, will attempt to infer ID from filepath using the DCASE sound event classes.
            spatial_velocity: Speed of a moving sound event in metres-per-second
            spatial_resolution: Resolution of a moving sound event in Hz (i.e., number of IRs created per second)
            shape: the shape of a moving event trajectory; one of "linear", "semicircular", "random", "sine", "sawtooth", "predefined"
            max_place_attempts (Numeric): the number of times to try and place an Event before giving up.
            event_kwargs: additional keyword arguments passed to Event.__init__

        Returns:
            the Event object added to the Scene

        Examples:
            Creating an event with random sampling of parameters.
            Here, note that "scene_start", "event_start", "duration", "snr" will be sampled at random
            from the distributions defined when initialising the `Scene`.

            >>> scene = Scene(...)
            >>> scene.add_event(
            >>>     event_type="static",
            >>>     filepath="some/path.wav"
            >>> )

            Creating an event with a predefined position:

            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     event_type="static",
            ...     filepath="some/path.wav",
            ...     alias="tester",
            ...     position=[-0.5, -0.5, 0.5],
            ...     polar=False,
            ...     ensure_direct_path=False
            ... )

            Creating an event with overrides:

            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     event_type="moving",
            ...     filepath="some/path.wav",
            ...     alias="tester",
            ...     event_start=5.0,
            ...     duration=5.0,
            ...     snr=0.0,
            ... )

        """
        # Call the requisite function to add the event
        if event_type == "static":
            event = self.add_event_static(
                filepath=filepath,
                alias=alias,
                position=position,
                mic=mic,
                polar=polar,
                ensure_direct_path=ensure_direct_path,
                scene_start=scene_start,
                event_start=event_start,
                duration=duration,
                snr=snr,
                class_id=class_id,
                class_label=class_label,
                augmentations=augmentations,
                max_place_attempts=max_place_attempts,
                **event_kwargs,
            )

        elif event_type == "moving":
            event = self.add_event_moving(
                filepath=filepath,
                alias=alias,
                position=position,
                polar=polar,
                mic=mic,
                shape=shape,
                scene_start=scene_start,
                event_start=event_start,
                duration=duration,
                snr=snr,
                class_id=class_id,
                class_label=class_label,
                spatial_resolution=spatial_resolution,
                spatial_velocity=spatial_velocity,
                augmentations=augmentations,
                ensure_direct_path=ensure_direct_path,
                max_place_attempts=max_place_attempts,
                **event_kwargs,
            )

        elif event_type == "predefined":
            if spatial_velocity is not None or spatial_resolution is not None:
                logger.warning(
                    "Predefined event will ignore `spatial_velocity` or `spatial_resolution` parameters"
                )
            event = self.add_event_predefined(
                filepath=filepath,
                trajectory=trajectory,
                alias=alias,
                augmentations=augmentations,
                scene_start=scene_start,
                event_start=event_start,
                duration=duration,
                snr=snr,
                class_id=class_id,
                class_label=class_label,
                ensure_direct_path=ensure_direct_path,
                max_place_attempts=max_place_attempts,
            )

        else:
            raise ValueError(
                f"Cannot parse event type {event_type}, expected either 'static', 'moving', or 'predefined'!"
            )

        # Log the creation of the event
        logger.info(f"Event added successfully: {event}")
        return event

    def _validate_user_defined_audio_filepath(
        self, user_filepath: Path, user_class_id: int
    ) -> None:
        """
        Validates the user defined audio filepath parameter.
        """
        # If we don't want to allow for duplicate filepaths, check this now
        if not self.allow_duplicate_audios:
            seen_audios = self._get_used_audios()
            if user_filepath in seen_audios:
                raise ValueError(
                    f"Audio file {str(user_filepath.resolve())} has already been added to the Scene. "
                    f"Either increase the number of `fg_paths` in Scene.__init__, "
                    f"choose a different audio file, "
                    f"or set `Scene.allow_duplicate_audios=True`."
                )

        # If we don't want to allow for duplicate class IDs, check this too
        if not self.allow_same_class_events:
            seen_classes = self._get_used_class_ids()
            # Try and resolve the class ID
            resolved_id = (
                self.class_mapping.infer_label_idx_from_filepath(user_filepath)[0]
                if user_class_id is None
                else user_class_id
            )
            if resolved_id in seen_classes:
                raise ValueError(
                    f"Audio file {str(user_filepath.resolve())} uses a class that has"
                    f" already been added to the Scene ({resolved_id}). "
                    f"Either choose a different audio file, "
                    f"or set `Scene.allow_same_class_events=True`."
                )

    def add_event_static(
        self,
        filepath: Optional[Union[str, Path]] = None,
        alias: Optional[str] = None,
        augmentations: Optional[
            Union[
                Iterable[Type[EventAugmentation]],
                Type[EventAugmentation],
                custom_types.Numeric,
            ]
        ] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        mic: Optional[str] = None,
        polar: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        scene_start: Optional[custom_types.Numeric] = None,
        event_start: Optional[custom_types.Numeric] = None,
        duration: Optional[custom_types.Numeric] = None,
        snr: Optional[custom_types.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
        **event_kwargs,
    ) -> Event:
        """
        Add a static event to the foreground with optional overrides.

        Note that the arguments "scene_start", "event_start", "duration", & "snr" will (by default) sample from their
        respective distributions, provided in `Scene.__init__`. If a numeric value is provided, this will be treated as
        an override and used instead of random sampling.

        Arguments:
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            augmentations: augmentation objects to associate with the Event.
                If a list of EventAugmentation objects or a single EventAugmentation object, these will be passed directly.
                If a number, this many augmentations will be sampled from either `Scene.event_augmentations`, or a master
                list of valid augmentations (defined inside `audiblelight.augmentations`)
                If not provided, EventAugmentations can be registered later by calling `register_augmentations` on the Event.
            position: Location to add the event. When not provided, a random point inside the mesh will be chosen.
            mic: String reference to a microphone inside `self.state.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            polar: When True, expects `position` to be provided in [azimuth, elevation, radius] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
            duration: Time the Event audio lasts in seconds. Must be a positive number.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            class_label: Optional label to use for sound event class.
                If not provided, will attempt to infer label from filepath using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, will attempt to infer ID from filepath using the DCASE sound event classes.
            max_place_attempts (Numeric): the number of times to try and place an Event before giving up.

        Returns:
            the Event object added to the Scene
        """
        # Get a default alias and a random filepath if these haven't been provided
        alias = (
            utils.get_default_alias("event", self.events) if alias is None else alias
        )

        # Check filepath over
        if filepath is not None:
            # Sanitise the filepath (check it exists on disk, etc.)
            filepath = utils.sanitise_filepath(filepath)

            # Run validation checks on the filepath
            self._validate_user_defined_audio_filepath(filepath, class_id)

        # Convert polar positions to cartesian here
        if polar:
            position = self._coerce_polar_position(position, mic)
            # Set mic to None so we don't add the offset again inside WorldState
            # TODO: this is hacky, could be better
            mic = None

        # Sample N random augmentations from our list, if required
        if isinstance(augmentations, custom_types.Numeric):
            augmentations = self._get_n_random_event_augmentations(augmentations)

        # Construct kwargs dictionary for emitter and event
        event_kwargs_full = dict(
            filepath=filepath,
            alias=alias,
            scene_start=scene_start,
            event_start=event_start,
            duration=duration,
            snr=snr,
            sample_rate=self.sample_rate,
            class_id=class_id,
            class_label=class_label,
            # No spatial resolution/velocity for static events
            spatial_resolution=None,
            spatial_velocity=None,
            # Shape is always "static"
            shape="static",
            augmentations=augmentations,
            # Emitter kwargs
            position=position,
            mic=mic,
            ensure_direct_path=ensure_direct_path,
            keep_existing=True,
            max_place_attempts=max_place_attempts,
            class_mapping=self.class_mapping,
            **event_kwargs,
        )

        # Try and create the event: returns True if placed, False if not
        placed = self._try_add_event(**event_kwargs_full)

        # Raise an error if we can't place the event correctly
        if not placed:
            raise ValueError(
                f"Could not place event in the mesh after {config.MAX_PLACE_ATTEMPTS} attempts. "
                f"Consider increasing the value of `max_overlap` (currently {self.max_overlap}) or the "
                f"`duration` of the scene (currently {self.duration})."
            )

        # Return the event with emitters already registered
        return self.get_event(alias)

    def add_event_moving(
        self,
        filepath: Optional[Union[str, Path]] = None,
        alias: Optional[str] = None,
        augmentations: Optional[
            Union[
                Iterable[Type[EventAugmentation]],
                Type[EventAugmentation],
                custom_types.Numeric,
            ]
        ] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        mic: Optional[str] = None,
        polar: Optional[bool] = False,
        shape: Optional[str] = None,
        scene_start: Optional[custom_types.Numeric] = None,
        event_start: Optional[custom_types.Numeric] = None,
        duration: Optional[custom_types.Numeric] = None,
        snr: Optional[custom_types.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        spatial_resolution: Optional[custom_types.Numeric] = None,
        spatial_velocity: Optional[custom_types.Numeric] = None,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
        **event_kwargs,
    ) -> Event:
        """
        Add a moving event to the foreground with optional overrides.

        Note that the arguments "scene_start", "event_start", "duration", "snr", "spatial_velocity", &
        "spatial_resolution" will (by default) sample from their respective distributions, provided in `Scene.__init__`.
        If a numeric value is provided, this will be treated as an override and used instead of random sampling.

        Arguments:
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            augmentations: augmentation objects to associate with the Event.
                If a list of EventAugmentation objects or a single EventAugmentation object, these will be passed directly.
                If a number, this many augmentations will be sampled from either `Scene.event_augmentations`, or a master
                list of valid augmentations (defined inside `audiblelight.augmentations`)
                If not provided, EventAugmentations can be registered later by calling `register_augmentations` on the Event.
            position: Starting point for the event. When not provided, a random point inside the mesh will be chosen.
            mic: String reference to a microphone inside `self.state.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            polar: When True, expects `position` to be provided in [azimuth, elevation, radius] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
            duration: Time the Event audio lasts in seconds. Must be a positive number.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            class_label: Optional label to use for sound event class.
                If not provided, will attempt to infer label from filepath using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, will attempt to infer ID from filepath using the DCASE sound event classes.
            spatial_velocity: Speed of a moving sound event in metres-per-second
            spatial_resolution: Resolution of a moving sound event in Hz (i.e., number of IRs created per second)
            shape: the shape of a moving event trajectory; one of "linear", "semicircular", "random", "sine", "sawtooth"
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
            max_place_attempts (Numeric): the number of times to try and place an Event before giving up.
            event_kwargs: additional keyword arguments passed to Event.__init__

        Returns:
            the Event object added to the Scene
        """
        # Convert polar positions to cartesian here
        if polar:
            position = self._coerce_polar_position(position, mic)

        # Get a default alias and a random filepath if these haven't been provided
        alias = (
            utils.get_default_alias("event", self.events) if alias is None else alias
        )

        # Check filepath over
        if filepath is not None:
            # Sanitise the filepath (check it exists on disk, etc.)
            filepath = utils.sanitise_filepath(filepath)

            # Run validation checks on the filepath
            self._validate_user_defined_audio_filepath(filepath, class_id)

        # Sample N random augmentations from our list, if required
        if isinstance(augmentations, custom_types.Numeric):
            augmentations = self._get_n_random_event_augmentations(augmentations)

        # Sample a random shape if not provided
        if shape is None:
            shape = random.choice(config.MOVING_EVENT_SHAPES)

        # Set up the kwargs dictionaries for the `define_trajectory` and `Event.__init__` funcs
        event_kwargs_full = dict(
            filepath=filepath,
            alias=alias,
            scene_start=scene_start,
            event_start=event_start,
            duration=duration,
            snr=snr,
            # Useful to store the shape of the moving event trajectory
            shape=shape,
            sample_rate=self.sample_rate,
            class_id=class_id,
            class_label=class_label,
            spatial_resolution=spatial_resolution,
            spatial_velocity=spatial_velocity,
            augmentations=augmentations,
            starting_position=position,
            ensure_direct_path=ensure_direct_path,
            max_place_attempts=max_place_attempts,
            class_mapping=self.class_mapping,
            **event_kwargs,
        )

        # Create the event with required arguments
        placed = self._try_add_event(**event_kwargs_full)

        # Raise an error if we can't place the event correctly
        if not placed:
            raise ValueError(
                f"Could not place event in the mesh after {config.MAX_PLACE_ATTEMPTS} attempts. "
                f"Consider increasing the value of `max_overlap` (currently {self.max_overlap}) or the "
                f"`duration` of the scene (currently {self.duration})."
            )

        # Return the event with emitters already registered
        return self.get_event(alias)

    # noinspection PyProtectedMember
    def _try_add_predefined_event(
        self,
        trajectory: Optional[np.ndarray],
        ensure_direct_path: Optional[bool],
        max_place_attempts: Optional[custom_types.Numeric],
        **event_kwargs,
    ) -> bool:
        """
        Tries to add an Event with given kwargs and predefined trajectory.

        The idea here is:
            - Iterate over ALL valid trajectories associated with the state, or a SINGLE user trajectory
                - Randomly sample duration, event start, etc. (or use overrides)
                - Grab spatial resolution + velocity from the trajectory in combination with the duration
                - If the trajectory is valid (e.g., has path to mic, inbounds with mesh), use it
                - Otherwise, resample the parameters and try again
            - If no valid combination for one trajectory, try with the next one

        Returns:
            bool: True if successful, False otherwise.
        """

        # Grab the alias: this should always be present inside the dictionary
        alias = event_kwargs["alias"]

        # Use only 1 placement attempt if all overrides are present
        has_overrides = all(
            k is not None in event_kwargs
            for k in ("scene_start", "event_start", "duration")
        )
        max_place_attempts_per_trajectory = (
            max_place_attempts if not has_overrides else 1
        )

        # Use only 1 placement attempt if trajectory provided
        if trajectory is not None:
            if not self.state._validate_position(trajectory):
                raise ValueError("Provided trajectory is invalid")
            trajectories = [trajectory]
        else:
            trajectories = self.state.waypoints

        # Pre-resolve all user-specified override values (only done once)
        #  Spatial resolution + velocity are defined by trajectory, not user
        overrides = {
            "scene_start": event_kwargs.get("scene_start"),
            "event_start": event_kwargs.get("event_start"),
            "duration": event_kwargs.get("duration"),
            "snr": event_kwargs.get("snr"),
        }

        # Get valid aliases from the input
        ensure_direct_path_to_mic = self.state._parse_valid_microphone_aliases(
            ensure_direct_path
        )

        # Iterate over all the trajectories
        #  If a user defined trajectory, we will only iterate once
        #  Otherwise, we iterate over all valid trajectories associated with the state
        for trajectory_current in trajectories:

            # Compute some statistics based on the duration
            n_points = trajectory_current.shape[0]
            start = trajectory_current[0]
            differences = trajectory_current[1:] - start
            distances = np.linalg.norm(differences, axis=1)
            max_distance = distances[np.argmax(distances)]

            # If required, check that a direct path exists to all mic objects from this trajectory
            for d in ensure_direct_path_to_mic:
                if not all(
                    self.state.path_exists_between_points(
                        t, self.get_microphone(d).coordinates_center
                    )
                    for t in trajectory_current
                ):
                    continue

            # Iterate a number of times over each trajectory
            #  This allows us e.g. to try different duration values for a single trajectory
            for _ in range(max_place_attempts_per_trajectory):

                # Copy once per attempt
                current_kws = event_kwargs.copy()

                # If we haven't passed in a duration override OR a distribution, default to using the full audio duration
                if overrides["duration"] is None and self.event_duration_dist is None:
                    current_kws["duration"] = None
                # Otherwise, try and sample from the distribution or use the override
                else:
                    current_kws["duration"] = utils.sample_distribution(
                        self.event_duration_dist, overrides["duration"]
                    )

                # Do the same for event start time
                if overrides["event_start"] is None and self.event_start_dist is None:
                    current_kws["event_start"] = None
                else:
                    current_kws["event_start"] = utils.sample_distribution(
                        self.event_start_dist, overrides["event_start"]
                    )

                # Sample values (with fallback to override if provided)
                current_kws.update(
                    {
                        "scene_start": utils.sample_distribution(
                            self.scene_start_dist, overrides["scene_start"]
                        ),
                        "snr": utils.sample_distribution(
                            self.snr_dist, overrides["snr"]
                        ),
                        "shape": "predefined",
                    }
                )

                # Create the event with the current keywords
                current_event = Event(**current_kws)

                # Reject this attempt if overlap would be exceeded
                if self._would_exceed_temporal_overlap(
                    current_event.scene_start, current_event.scene_end
                ):
                    continue

                # Extract the spatial resolution from the trajectory
                #  equivalent to number of points in trajectory over duration
                #  We need the duration sampled for the current iteration here
                #  So this has to be done inside the second loop
                spatial_resolution = (
                    utils.sanitise_positive_number(
                        n_points / current_event.duration, cast_to=round
                    )
                    - 1
                )
                current_event.spatial_resolution = spatial_resolution

                # Extract the spatial velocity from the trajectory
                #  equivalent to total distance travelled over duration
                spatial_velocity = max_distance / current_event.duration
                current_event.spatial_velocity = spatial_velocity

                if (
                    current_event.spatial_velocity > self.event_velocity_dist.max
                    or current_event.spatial_velocity < self.event_velocity_dist.min
                ):
                    continue

                # Store the event and register the emitters with the current trajectory
                self.state._add_emitters_without_validating(trajectory_current, alias)
                emitters = self.state.get_emitters(alias)
                if len(emitters) != len(trajectory_current):
                    self.clear_event(alias)
                    raise ValueError(
                        f"Did not add expected number of emitters into the WorldState "
                        f"(expected {len(trajectory_current)}, got {len(emitters)})"
                    )
                current_event.register_emitters(emitters)
                self.events[alias] = current_event
                return True

        return False

    # noinspection PyProtectedMember
    def add_event_predefined(
        self,
        filepath: Optional[Union[str, Path]] = None,
        trajectory: Optional[np.ndarray] = None,
        alias: Optional[str] = None,
        augmentations: Optional[
            Union[
                Iterable[Type[EventAugmentation]],
                Type[EventAugmentation],
                custom_types.Numeric,
            ]
        ] = None,
        scene_start: Optional[custom_types.Numeric] = None,
        event_start: Optional[custom_types.Numeric] = None,
        duration: Optional[custom_types.Numeric] = None,
        snr: Optional[custom_types.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
    ):
        """
        Add a moving event to the foreground that follows a predefined path.

        The spatial velocity and resolution of the event will be inferred from the trajectory itself, in combination
        with the duration (which may be provided or randomly sampled).
        """
        # Get a default alias and a random filepath if these haven't been provided
        alias = (
            utils.get_default_alias("event", self.events) if alias is None else alias
        )
        filepath = (
            self._get_random_audio(self.fg_audios)
            if filepath is None
            else utils.sanitise_filepath(filepath)
        )

        # Check filepath over
        if filepath is not None:
            # Sanitise the filepath (check it exists on disk, etc.)
            filepath = utils.sanitise_filepath(filepath)

            # Run validation checks on the filepath
            self._validate_user_defined_audio_filepath(filepath, class_id)

        # Sample N random augmentations from our list, if required
        if isinstance(augmentations, custom_types.Numeric):
            augmentations = self._get_n_random_event_augmentations(augmentations)

        # If no movement trajectory provided, try and sample one from the state
        if not isinstance(trajectory, np.ndarray) and len(self.state.waypoints) == 0:
            raise ValueError(
                "State must have waypoints: did you set `waypoints_json` correctly?"
            )

        event_kwargs = dict(
            filepath=filepath,
            alias=alias,
            scene_start=scene_start,
            event_start=event_start,
            duration=duration,
            snr=snr,
            sample_rate=self.sample_rate,
            class_id=class_id,
            class_label=class_label,
            augmentations=augmentations,
            class_mapping=self.class_mapping,
        )
        # Pre-initialise the event with required arguments + register the emitters
        utils.validate_kwargs(Event.__init__, **event_kwargs)
        placed = self._try_add_predefined_event(
            **event_kwargs,
            trajectory=trajectory,
            max_place_attempts=max_place_attempts,
            ensure_direct_path=ensure_direct_path,
        )

        # Raise an error if we can't place the event correctly
        if not placed:
            # No need to clear out any emitters (as in `add_event_static`) because we haven't placed them yet
            raise ValueError(
                f"Could not place event in the mesh after {config.MAX_PLACE_ATTEMPTS} attempts. "
                f"Consider increasing the value of `max_overlap` (currently {self.max_overlap}) or the "
                f"`duration` of the scene (currently {self.duration})."
            )

        # Return the event we just created
        return self.get_event(alias)

    def _would_exceed_temporal_overlap(
        self, new_event_start: float, new_event_end: float
    ) -> bool:
        """
        Determine whether an event is overlapping with other events more than `max_overlap` times.
        """

        intersections = 0
        for event_alias, event in self.events.items():
            # Check if intervals [new_start, new_end] and [existing_start, existing_end] overlap
            if new_event_start < event.scene_end and new_event_end > event.scene_start:
                intersections += 1
        return intersections >= self.max_overlap

    # noinspection PyProtectedMember
    # noinspection PyUnboundLocalVariable
    def generate(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        audio: bool = True,
        metadata_json: bool = True,
        metadata_dcase: bool = True,
        audio_fname: Optional[Union[str, Path]] = "audio_out",
        metadata_fname: Optional[Union[str, Path]] = "metadata_out",
    ) -> None:
        """
        Render scene to disk. Currently only audio and metadata are rendered.

        Arguments:
            output_dir: directory to save the output, defaults to a temp directory inside AudibleLight/spatial_scenes
            audio: whether to save audio as an output, default to `True`
            metadata_json: whether to save metadata JSON file, default to `True`
            metadata_dcase: whether to save metadata CSVs in DCASE format, default to `True`
            audio_fname: name to use for the output audio file, default to "audio_out"
            metadata_fname: name to use for the output metadata, default to "metadata_out"

        Returns:
            None
        """
        from audiblelight.synthesize import (
            generate_dcase2024_metadata,
            generate_scene_audio_from_events,
            render_audio_for_all_scene_events,
        )

        # Sanitise output directory
        #  Create a new temporary directory inside project root if not provided
        if output_dir is None:
            output_dir = Path.cwd()
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist")

        # Sanitise filepaths: strip out suffixes, we'll add these in later
        audio_path = (output_dir / audio_fname).with_suffix("")
        metadata_path = (output_dir / metadata_fname).with_suffix("")

        # Render all the audio
        #  This renders the IRs inside the worldstate
        #  It then populates the `.spatial_audio` attribute inside each Event
        #  And populates the `audio` attribute inside this instance
        render_audio_for_all_scene_events(self)
        generate_scene_audio_from_events(self)

        # Write the audio output to a separate .wav, one per mic
        if audio:
            for mic_alias, mic_audio in self.audio.items():
                sf.write(
                    audio_path.with_suffix(".wav").with_stem(
                        f"{audio_path.name}_{mic_alias}"
                    ),
                    mic_audio.T,
                    int(self.sample_rate),
                )

        # Get the metadata and add the spatial audio format in
        if metadata_json or metadata_dcase:
            metadata = self.to_dict()

        # Dump the metadata to a JSON
        if metadata_json:
            with open(metadata_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

        # Generate DCASE-2024 style metadata
        if metadata_dcase:
            dcase_meta = generate_dcase2024_metadata(self)
            # Save a single CSV file for every microphone we have
            for mic, df in dcase_meta.items():
                outp = metadata_path.with_suffix(".csv").with_stem(
                    f"{metadata_path.name}_{mic}"
                )
                df.to_csv(outp, sep=",", encoding="utf-8", header=None)

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        return dict(
            audiblelight_version=version("audiblelight"),
            rlr_audio_propagation_version=version("rlr_audio_propagation"),
            creation_time=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            duration=self.duration,
            backend=self.state.name,
            sample_rate=self.sample_rate,
            ref_db=self.ref_db,
            max_overlap=self.max_overlap,
            fg_path=[str(fg.resolve()) for fg in self.fg_paths],
            bg_path=[str(fg.resolve()) for fg in self.bg_paths],
            ambience={k: a.to_dict() for k, a in self.ambience.items()},
            events={k: e.to_dict() for k, e in self.events.items()},
            state=self.state.to_dict(),
            class_mapping=(
                self.class_mapping.to_dict() if self.class_mapping is not None else None
            ),
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate a `Scene` from a dictionary.

        The new `Scene` will have the same WorldState, Emitters, Events, and Microphones as the original, serialised
        dictionary created from `to_dict`. Ensure that any necessary files (e.g. meshes, audio files) are located in
        the same places as specified in the dictionary.

        Note that, currently, distribution objects (e.g., `Scene.event_start_dist`) cannot be loaded from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `Scene`.

        Returns:
            Scene instance.
        """

        # Sanitise the input
        for expected in [
            "audiblelight_version",
            "rlr_audio_propagation_version",
            "duration",
            "ref_db",
            "ambience",
            "events",
            "state",
            "sample_rate",
            "backend",
            "class_mapping",
        ]:
            if expected not in input_dict:
                raise KeyError("Missing key: '{}'".format(expected))

        # Raise a warning on a version mismatch for both audiblelight and rlr_audio_propagation
        loaded_version = input_dict["audiblelight_version"]
        act_version = version("audiblelight")
        if loaded_version != act_version:
            logger.error(
                f"This Scene appears to have been created using a different version of `AudibleLight`. "
                f"The currently installed version is v.{act_version}, but the Scene was created "
                f"with v.{loaded_version}. AudibleLight will attempt to load the Scene; but if you encounter "
                f"errors, you should try running `pip install audiblelight=={act_version}`"
            )

        loaded_rlr = input_dict["rlr_audio_propagation_version"]
        actual_rlr = version("rlr_audio_propagation")
        if loaded_rlr != actual_rlr:
            logger.error(
                f"This Scene appears to have been created using a different version of `rlr_audio_propagation`"
                f". The currently installed version is v.{actual_rlr}, but the Scene was created "
                f"with v.{loaded_rlr}. AudibleLight will attempt to load the Scene; but if you encounter "
                f"errors, you should try running `pip install rlr_audio_propagation=={loaded_rlr}`"
            )

        # Instantiate the scene
        #  TODO: figure out some way to handle loading distributions here (non trivial as Scipy distributions cannot
        #   easily be saved to disk)
        logger.warning(
            "Currently, distributions cannot be loaded with `Scene.from_dict`. You will need to manually "
            "redefine these using, for instance, setattr(scene, 'event_start_dist', ...), repeating this "
            "for every distribution."
        )

        # Instantiate the state, which also creates all the emitters and microphones
        state = WorldState.from_dict(input_dict["state"])

        # Instantiate the class ampping
        class_mapping = ClassMapping.from_dict(input_dict["class_mapping"])

        # Pass the backend directly in when creating the Scene
        instantiated_scene = cls(
            duration=input_dict["duration"],
            backend=state,
            sample_rate=input_dict["sample_rate"],
            fg_path=input_dict["fg_path"],
            bg_path=input_dict["bg_path"],
            ref_db=input_dict["ref_db"],
            max_overlap=input_dict["max_overlap"],
            class_mapping=class_mapping,
        )

        # Instantiate the events by iterating over the list
        instantiated_scene.events = OrderedDict(
            {k: Event.from_dict(v) for k, v in input_dict["events"].items()}
        )

        # Instantiate the ambience in the same way
        instantiated_scene.ambience = OrderedDict(
            {k: Ambience.from_dict(v) for k, v in input_dict["ambience"].items()}
        )

        return instantiated_scene

    @classmethod
    def from_json(cls, json_fpath: Union[str, Path]):
        """
        Instantiate a `Scene` from a JSON file.

        Arguments:
            json_fpath: Path to the JSON file to load.

        Returns:
            Scene instance.
        """

        # Sanitise the filepath to a Path object
        sanitised_path = utils.sanitise_filepath(json_fpath)

        # Load the JSON to a dictionary
        with open(sanitised_path, "r") as f:
            loaded = json.load(f)

        # Use our existing function to load the dictionary
        return cls.from_dict(loaded)

    def get_events(self) -> list[Event]:
        """
        Return a list of all events for this scene, as in `self.events.values()`
        """
        return list(self.events.values())

    # noinspection PyUnreachableCode
    def get_event(self, alias_or_idx: Union[str, int]) -> Event:
        """
        Given a valid alias, get an associated event either by alias (string) or idx (int).
        """
        # Trying to get the event by its alias
        if isinstance(alias_or_idx, str):
            if alias_or_idx in self.events.keys():
                return self.events[alias_or_idx]
            else:
                raise KeyError("Event alias '{}' not found.".format(alias_or_idx))

        # Trying to get the event by its index
        elif isinstance(alias_or_idx, int):
            try:
                return list(self.events.values())[alias_or_idx]
            except IndexError:
                raise IndexError("No event with index {}.".format(alias_or_idx))

        # We don't know how to get the event
        else:
            raise TypeError(
                "Expected `str` or `int` but got {}".format(type(alias_or_idx))
            )

    def get_emitters(self, alias: str) -> list[Emitter]:
        """
        Alias for `WorldState.get_emitters`
        """
        return self.state.get_emitters(alias)

    def get_emitter(self, alias: str, emitter_idx: int = 0) -> Emitter:
        """
        Alias for `WorldState.get_emitter`
        """
        return self.state.get_emitter(alias, emitter_idx)

    def get_microphone(self, alias: str) -> Type["MicArray"]:
        """
        Alias for `WorldState.get_microphone`
        """
        return self.state.get_microphone(alias)

    def get_ambience(self, alias) -> Ambience:
        """
        Given a valid alias, get an associated ambience event, as in `self.ambience[alias]`
        """
        if alias in self.ambience.keys():
            return self.ambience[alias]
        else:
            raise KeyError("Ambience alias '{}' not found.".format(alias))

    def get_ambiences(self) -> list[Ambience]:
        """
        Get all ambience objects, as in `self.ambience.values()`
        """
        return list(self.ambience.values())

    def get_class_mapping(self) -> Type[TClassMapping]:
        """
        Alias for `ClassMapping.mapping`
        """
        if self.class_mapping is not None:
            return self.class_mapping.to_dict()
        else:
            return None

    # noinspection PyProtectedMember
    def clear_events(self) -> None:
        """
        Removes all current events and emitters from the state
        """
        self.events = OrderedDict()
        self.state.clear_emitters()

    # noinspection PyProtectedMember
    def clear_event(self, alias: str) -> None:
        """
        Given an alias for an event, clears the event and updates the state.

        Note: simply calling `del self.events[alias]` is not enough; we also need to remove the source from the
        ray-tracing engine by updating the `state.emitters` dictionary and calling `state._update`.
        """
        if alias in self.events.keys():
            # Grab the event
            ev = self.events[alias]
            # Clear out all emitters associated with this event
            #  This all calls `state._update` to remove them from the backend
            for emitter in ev.get_emitters():
                self.state.clear_emitter(emitter.alias)
            # Remove the event from the Scene API
            del self.events[alias]
        else:
            raise KeyError("Event alias '{}' not found.".format(alias))

    def clear_emitters(self) -> None:
        """
        Alias for `WorldState.clear_emitters`.
        """
        # Raise a warning when we might orphan events
        if len(self.events) > 0:
            logger.warning(
                "Clearing emitters from a scene may orphan its associated events. It is recommended to "
                "call `Scene.clear_events()` to clear all events and their associated emitters, "
                "rather than this function."
            )
        self.state.clear_emitters()

    def clear_microphones(self) -> None:
        """
        Alias for `WorldState.clear_microphones`.
        """
        self.state.clear_microphones()

    def clear_emitter(self, alias: str) -> None:
        """
        Alias for `WorldState.clear_emitter`.
        """
        # Raise a warning when we might orphan an event
        if len(self.events) > 0 and alias in self.events:
            logger.warning(
                f"Clearing emitters with the alias '{alias}' will orphan an event. It is recommended to "
                f"instead call `Scene.clear_event(alias)` to remove this event and its associated emitters, "
                f"rather than calling this function."
            )
        self.state.clear_emitter(alias)

    def clear_microphone(self, alias: str) -> None:
        """
        Alias for `WorldState.clear_microphone`.
        """
        self.state.clear_microphone(alias)

    def clear_ambience(self) -> None:
        """
        Removes all current ambience events.
        """
        self.ambience = OrderedDict()
