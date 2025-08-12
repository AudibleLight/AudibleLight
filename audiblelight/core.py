#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core modules and functions for generation and synthesis."""

import json
import os
import random
from collections import OrderedDict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator, Optional, Type, Union

import numpy as np
import soundfile as sf
from deepdiff import DeepDiff
from loguru import logger
from scipy import stats

from audiblelight import __version__, utils
from audiblelight.ambience import Ambience
from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import Emitter, WorldState

MAX_OVERLAPPING_EVENTS = 3
REF_DB = -50
WARN_WHEN_DURATION_LOWER_THAN = 5


class Scene:
    def __init__(
        self,
        duration: utils.Numeric,
        mesh_path: Union[str, Path],
        fg_path: Optional[Union[str, Path]] = None,
        state_kwargs: Optional[dict] = None,
        ref_db: Optional[utils.Numeric] = REF_DB,
        scene_start_dist: Optional[utils.DistributionLike] = None,
        event_start_dist: Optional[utils.DistributionLike] = None,
        event_duration_dist: Optional[utils.DistributionLike] = None,
        event_velocity_dist: Optional[utils.DistributionLike] = None,
        event_resolution_dist: Optional[utils.DistributionLike] = None,
        snr_dist: Optional[utils.DistributionLike] = None,
        max_overlap: Optional[utils.Numeric] = MAX_OVERLAPPING_EVENTS,
    ):
        # Set attributes passed in by the user
        self.duration = utils.sanitise_positive_number(duration)
        # Raise a warning when the duration is very short.
        if self.duration < WARN_WHEN_DURATION_LOWER_THAN:
            logger.warning(
                f"The duration for this Scene is very short ({duration:.2f} seconds). "
                f"You may encounter issues with Events overlapping or being truncated to fit the "
                f"duration of the Scene. It is recommended to increase the duration to at least "
                f"{WARN_WHEN_DURATION_LOWER_THAN} seconds."
            )

        self.fg_path = (
            utils.sanitise_directory(fg_path) if fg_path is not None else None
        )
        self.ref_db = utils.sanitise_ref_db(ref_db)
        # Time overlaps (we could include a space overlaps parameter too)
        self.max_overlap = int(utils.sanitise_positive_number(max_overlap))

        # Instantiate the `WorldState` object, which loads the mesh and sets up the ray-tracing engine
        if state_kwargs is None:
            state_kwargs = {}
        utils.validate_kwargs(WorldState.__init__, **state_kwargs)
        self.state = WorldState(mesh_path, **state_kwargs)

        self.sample_rate = self.state.ctx.config.sample_rate

        # Grab some attributes from the WorldState to make them easier to access
        self.mesh = self.state.mesh
        # self.irs = self.state.irs

        # Define defaults for all distributions
        #  Events can start any time within the duration of the scene, minus some padding
        if scene_start_dist is None:
            scene_start_dist = stats.uniform(0.0, self.duration - 1)
        #  Events move between 0.25 and 2.0 metres per second
        if event_velocity_dist is None:
            event_velocity_dist = stats.uniform(0.25, 2.0)
        #  Events have a resolution of between 1-4 Hz (i.e., number of IRs per second)
        if event_resolution_dist is None:
            event_resolution_dist = stats.uniform(1.0, 4.0)
        #  Events have an SNR with a mean of 5, SD of 1, and boundary between 2 and 8
        if snr_dist is None:
            snr_dist = stats.truncnorm(a=-3, b=3, loc=5, scale=1)

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

        # Assuming path structure with audio files organized in directories per category of interest
        self.fg_category_paths = (
            utils.list_deepest_directories(self.fg_path)
            if self.fg_path is not None
            else None
        )

        self.events = OrderedDict()

        # Background noise
        #  if not None (i.e., with a call to `add_ambience`), will be added to audio when synthesising
        self.ambience = OrderedDict()

        self.audio = None

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
        noise: Optional[Union[str, utils.Numeric]] = None,
        channels: Optional[int] = None,
        ref_db: Optional[utils.Numeric] = None,
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
            filepath (str or Path): a path to an audio file on the disk. Must be provided when `noise` is None.
            noise (str): either the type of noise to generate, e.g. "white", "red", or an arbitrary numeric exponent to
                use when generating noise with `powerlaw_psd_gaussian`. Must be provided if `filepath` is None.
            ref_db (Numeric): the noise floor, in decibels
            alias (str): string reference to refer to this `Ambience` object inside `Scene.ambience`
            kwargs: additional keyword arguments passed to `audiblelight.ambience.powerlaw_psd_gaussian`
        """
        # If the number of channels is not provided, try and get this from the number of microphone capsules
        if channels is None:
            available_mics = [mic.n_capsules for mic in self.state.microphones.values()]
            # Raise an error when added microphones have a different number of channels
            if not all([a == available_mics[0] for a in available_mics]):
                raise TypeError(
                    "Cannot infer noise channels when available microphones have different number of capsules"
                )
            else:
                channels = available_mics[0]

        # Get the alias for this ambience event: either default or user-provided
        alias = (
            utils.get_default_alias("ambience", self.ambience)
            if alias is None
            else alias
        )
        if alias in self.ambience:
            raise KeyError(
                f"Ambience event with alias {alias} has already been added to the scene!"
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
        max_place_attempts = utils.MAX_PLACE_ATTEMPTS if not has_overrides else 1

        # Pre-resolve all user-specified override values (only done once)
        overrides = {
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
            current_event = Event(**current_kws)

            # Reject this attempt if overlap would be exceeded
            if self._would_exceed_temporal_overlap(
                current_event.scene_start, current_event.duration
            ):
                continue

            # Store the event: we'll register the emitters later in the function
            self.events[alias] = current_event
            return True

        return False

    def _get_random_foreground_audio(self) -> Path:
        """
        Gets a path to a random foreground audio file from the provided directory
        """
        if self.fg_category_paths is None:
            raise ValueError("No foreground audio path specified!")
        audios = []
        for fg_category_path in self.fg_category_paths:
            for i_ in os.listdir(fg_category_path):
                if i_.endswith(utils.AUDIO_EXTS):
                    audios.append(fg_category_path / Path(i_))
        if len(audios) == 0:
            raise FileNotFoundError("No audio files found!")
        return utils.sanitise_filepath(random.choice(audios))

    def add_event(
        self,
        event_type: Optional[str] = "static",
        filepath: Optional[Union[str, Path]] = None,
        alias: Optional[str] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        mic: Optional[str] = None,
        polar: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        scene_start: Optional[utils.Numeric] = None,
        event_start: Optional[utils.Numeric] = None,
        duration: Optional[utils.Numeric] = None,
        snr: Optional[utils.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        shape: Optional[str] = None,
        spatial_resolution: Optional[utils.Numeric] = None,
        spatial_velocity: Optional[utils.Numeric] = None,
    ) -> Event:
        """
        Add an event to the foreground, either "static" or "moving"

        Arguments:
            event_type (str): the type of event to add, must be either "static" or "moving"
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            position: Location to add the event.
                When `event_type=="static"`, this will be the position of the Event.
                When `event_type=="moving"`, this will be the starting position of the Event.
                When not provided, a random point inside the mesh will be chosen.
            mic: String reference to a microphone inside `self.state.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            polar: When True, expects `position` to be provided in [azimuth, colatitude, elevation] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
                If not provided, defaults to the beginning of the Scene (i.e., 0 seconds).
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
                If not provided, defaults to starting the audio at the very beginning (i.e., 0 seconds).
            duration: Time the Event audio lasts in seconds. Must be a positive number.
                If None or greater than the duration of the audio, defaults to using the full duration of the audio.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            class_label: Optional label to use for sound event class.
                If not provided, the label will attempt to be inferred from the ID using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, the ID will attempt to be inferred from the label using the DCASE sound event classes.
            spatial_velocity: Speed of a moving sound event in metres-per-second
            spatial_resolution: Resolution of a moving sound event in Hz (i.e., number of IRs created per second)
            shape: the shape of a moving event trajectory; must be one of "linear", "circular", "random".

        Returns:
            the Event object added to the Scene

        Examples:
            Creating an event with a predefined position
            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     event_type="static",
            ...     filepath=...,
            ...     alias="tester",
            ...     position=[-0.5, -0.5, 0.5],
            ...     polar=False,
            ...     ensure_direct_path=False
            ... )

            Creating an event with overrides
            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     event_type="moving",
            ...     filepath=...,
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
            )

        elif event_type == "moving":
            event = self.add_event_moving(
                filepath=filepath,
                alias=alias,
                position=position,
                shape=shape,
                scene_start=scene_start,
                event_start=event_start,
                duration=duration,
                snr=snr,
                class_id=class_id,
                class_label=class_label,
                spatial_resolution=spatial_resolution,
                spatial_velocity=spatial_velocity,
            )

        else:
            raise ValueError(
                f"Cannot parse event type {event_type}, expected either 'static' or 'moving'!"
            )

        # Log the creation of the event
        logger.info(f"Event added successfully: {event}")
        return event

    def add_event_static(
        self,
        filepath: Optional[Union[str, Path]] = None,
        alias: Optional[str] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        mic: Optional[str] = None,
        polar: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        scene_start: Optional[utils.Numeric] = None,
        event_start: Optional[utils.Numeric] = None,
        duration: Optional[utils.Numeric] = None,
        snr: Optional[utils.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
    ) -> Event:
        """
        Add a static event to the foreground with optional overrides.

        Arguments:
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            position: Location to add the event.
                When `event_type=="static"`, this will be the position of the Event.
                When `event_type=="moving"`, this will be the starting position of the Event.
                When not provided, a random point inside the mesh will be chosen.
            mic: String reference to a microphone inside `self.state.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            polar: When True, expects `position` to be provided in [azimuth, colatitude, elevation] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
                If not provided, defaults to the beginning of the Scene (i.e., 0 seconds).
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
                If not provided, defaults to starting the audio at the very beginning (i.e., 0 seconds).
            duration: Time the Event audio lasts in seconds. Must be a positive number.
                If None or greater than the duration of the audio, defaults to using the full duration of the audio.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            class_label: Optional label to use for sound event class.
                If not provided, the label will attempt to be inferred from the ID using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, the ID will attempt to be inferred from the label using the DCASE sound event classes.

        Returns:
            the Event object added to the Scene
        """
        # Get a default alias and a random filepath if these haven't been provided
        alias = (
            utils.get_default_alias("event", self.events) if alias is None else alias
        )
        filepath = (
            self._get_random_foreground_audio()
            if filepath is None
            else utils.sanitise_filepath(filepath)
        )

        # Construct kwargs dictionary for emitter and event
        emitter_kwargs = dict(
            position=position,
            alias=alias,
            mic=mic,
            polar=polar,
            ensure_direct_path=ensure_direct_path,
            keep_existing=True,
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
            # No spatial resolution/velocity for static events
            spatial_resolution=None,
            spatial_velocity=None,
        )

        # Add the emitters associated with the event to the worldstate
        #  This will perform spatial logic checks for e.g. ensuring that the emitter won't collide with anything
        #  else inside the mesh, such as another emitter, microphone, or the mesh itself
        utils.validate_kwargs(self.state.add_emitter, **emitter_kwargs)
        self.state.add_emitter(**emitter_kwargs)

        # Try and create the event: returns True if placed, False if not
        utils.validate_kwargs(Event.__init__, **event_kwargs)
        placed = self._try_add_event(**event_kwargs)

        # Raise an error if we can't place the event correctly
        if not placed:
            # Need to tidy up the emitter we placed above to prevent it becoming an orphan
            self.clear_emitter(alias)
            raise ValueError(
                f"Could not place event in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                f"Consider increasing the value of `max_overlap`."
            )

        # Get emitters from internal state and register them with the event
        emitters = self.state.get_emitters(alias)
        event = self.get_event(alias)
        event.register_emitters(emitters)

        return event

    # noinspection PyProtectedMember
    def add_event_moving(
        self,
        filepath: Optional[Union[str, Path]] = None,
        alias: Optional[str] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        shape: Optional[str] = None,
        scene_start: Optional[utils.Numeric] = None,
        event_start: Optional[utils.Numeric] = None,
        duration: Optional[utils.Numeric] = None,
        snr: Optional[utils.Numeric] = None,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        spatial_resolution: Optional[utils.Numeric] = None,
        spatial_velocity: Optional[utils.Numeric] = None,
    ) -> Event:
        """
        Add a moving event to the foreground with optional overrides.

        Arguments:
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            position: Location to add the event.
                When `event_type=="static"`, this will be the position of the Event.
                When `event_type=="moving"`, this will be the starting position of the Event.
                When not provided, a random point inside the mesh will be chosen.
            scene_start: Time to start the Event within the Scene, in seconds. Must be a positive number.
                If not provided, defaults to the beginning of the Scene (i.e., 0 seconds).
            event_start: Time to start the Event audio from, in seconds. Must be a positive number.
                If not provided, defaults to starting the audio at the very beginning (i.e., 0 seconds).
            duration: Time the Event audio lasts in seconds. Must be a positive number.
                If None or greater than the duration of the audio, defaults to using the full duration of the audio.
            snr: Signal to noise ratio for the audio file with respect to the noise floor
            class_label: Optional label to use for sound event class.
                If not provided, the label will attempt to be inferred from the ID using the DCASE sound event classes.
            class_id: Optional ID to use for sound event class.
                If not provided, the ID will attempt to be inferred from the label using the DCASE sound event classes.
            spatial_velocity: Speed of a moving sound event in metres-per-second
            spatial_resolution: Resolution of a moving sound event in Hz (i.e., number of IRs created per second)
            shape: the shape of a moving event trajectory; must be one of "linear", "circular", "random".

        Returns:
            the Event object added to the Scene
        """
        # Get a default alias and a random filepath if these haven't been provided
        alias = (
            utils.get_default_alias("event", self.events) if alias is None else alias
        )
        filepath = (
            self._get_random_foreground_audio()
            if filepath is None
            else utils.sanitise_filepath(filepath)
        )

        # Set up the kwargs dictionaries for the `define_trajectory` and `Event.__init__` funcs
        emitter_kwargs = dict(
            starting_position=position,
            shape=shape,
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
            spatial_resolution=spatial_resolution,
            spatial_velocity=spatial_velocity,
        )

        # Pre-initialise the event with required arguments
        #  Note that this DOES NOT register the emitters.
        #  We simply need to get the sampled duration, etc., directly from the Event object
        utils.validate_kwargs(Event.__init__, **event_kwargs)
        placed = self._try_add_event(**event_kwargs)

        # Raise an error if we can't place the event correctly
        if not placed:
            # No need to clear out any emitters (as in `add_event_static`) because we haven't placed them yet
            raise ValueError(
                f"Could not place event in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                f"Consider increasing the value of `max_overlap`."
            )

        # Grab the event we just created
        event = self.get_event(alias)

        # Update the kwargs we'll use to create the trajectory with parameters from the event
        emitter_kwargs["duration"] = event.duration
        emitter_kwargs["velocity"] = event.spatial_velocity
        emitter_kwargs["resolution"] = event.spatial_resolution
        utils.validate_kwargs(self.state.define_trajectory, **emitter_kwargs)

        # Define the trajectory
        trajectory = self.state.define_trajectory(**emitter_kwargs)

        # Add the emitters to the state with the desired aliases
        #  This just adds the emitters in a loop with no additional checks
        #  We already perform these checks inside `define_trajectory`.
        self.state._add_emitters_without_validating(trajectory, alias)

        # Grab the emitters we just created and register them with the event
        emitters = self.state.get_emitters(alias)
        if len(emitters) != len(trajectory):
            raise ValueError(
                f"Did not add expected number of emitters into the WorldState "
                f"(expected {len(trajectory)}, got {len(emitters)})"
            )
        event.register_emitters(emitters)

        return event

    def _would_exceed_temporal_overlap(
        self, new_event_start: float, new_event_duration: float
    ) -> bool:
        """
        Determine whether an event is overlapping with other events more than `max_overlap` times.
        """

        intersections = 0
        new_event_end = new_event_start + new_event_duration
        for event_alias, event in self.events.items():
            # Check if intervals [new_start, new_end] and [existing_start, existing_end] overlap
            if new_event_start < event.scene_end and new_event_end > event.scene_start:
                intersections += 1
        return intersections >= self.max_overlap

    def generate(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        spatial_audio_format: Optional[str] = "A",
    ) -> None:
        """
        Render scene to disk. Currently only audio and metadata are rendered.

        Arguments:
            audio_path: Path to the audio file.
            metadata_path: Path to the metadata file.
            spatial_audio_format: Format to use for saving spatial audio, defaults to ambisonics "A" format

        Returns:
            None
        """
        from audiblelight.synthesize import (
            generate_scene_audio_from_events,
            render_audio_for_all_scene_events,
            validate_scene,
        )

        # Render all the audio
        #  This renders the IRs inside the worldstate
        #  It then populates the `.spatial_audio` attribute inside each Event
        #  And populates the `audio` attribute inside this instance
        validate_scene(self)
        render_audio_for_all_scene_events(self)
        generate_scene_audio_from_events(self)

        # Write the audio output
        sf.write(audio_path, self.audio.T, int(self.state.ctx.config.sample_rate))

        # Get the metadata and add the spatial audio format in
        metadata = self.to_dict()
        metadata["spatial_format"] = spatial_audio_format

        # Dump the metadata to a JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        return dict(
            audiblelight_version=__version__,
            rlr_audio_propagation_version=version("rlr_audio_propagation"),
            creation_time=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            duration=self.duration,
            ref_db=self.ref_db,
            max_overlap=self.max_overlap,
            fg_path=str(self.fg_path),
            ambience={k: a.to_dict() for k, a in self.ambience.items()},
            events={k: e.to_dict() for k, e in self.events.items()},
            state=self.state.to_dict(),
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
        ]:
            if expected not in input_dict:
                raise KeyError("Missing key: '{}'".format(expected))

        # Raise a warning on a version mismatch for both audiblelight and rlr_audio_propagation
        loaded_version = input_dict["audiblelight_version"]
        if loaded_version != __version__:
            logger.error(
                f"This Scene appears to have been created using a different version of `AudibleLight`. "
                f"The currently installed version is v.{__version__}, but the Scene was created "
                f"with v.{loaded_version}. AudibleLight will attempt to load the Scene; but if you encounter "
                f"errors, you should try running `pip install audiblelight=={__version__}`"
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
        instantiated_scene = cls(
            duration=input_dict["duration"],
            mesh_path=input_dict["state"]["mesh"]["fpath"],
            fg_path=input_dict["fg_path"],
            ref_db=input_dict["ref_db"],
            max_overlap=input_dict["max_overlap"],
        )

        # Instantiate the state, which also creates all the emitters and microphones
        instantiated_scene.state = WorldState.from_dict(input_dict["state"])

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
            del self.events[alias]
            self.state.clear_emitter(alias)  # this calls state._update for us
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
