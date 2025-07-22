#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core modules and functions for generation and synthesis."""

import os
import json
import random
from pathlib import Path
from typing import Union, Optional, Type, Any

import soundfile as sf
import numpy as np
from scipy import stats

from audiblelight.event import Event
from audiblelight.worldstate import WorldState, Emitter
from audiblelight.micarrays import MicArray
from audiblelight import utils


MAX_OVERLAPPING_EVENTS = 3
REF_DB = -50


class Scene:
    def __init__(
            self,
            duration: utils.Numeric,
            mesh_path: Union[str, Path],
            mic_arrays: Union[dict, list[dict], str, list[str], Type['MicArray'], list[Type['MicArray']]],
            fg_path: Optional[Union[str, Path]] = None,
            state_kwargs: Optional[dict] = None,
            ref_db: Optional[utils.Numeric] = REF_DB,
            event_start_dist: Optional[utils.DistributionLike] = None,
            event_duration_dist: Optional[utils.DistributionLike] = None,
            event_velocity_dist: Optional[utils.DistributionLike] = None,
            event_resolution_dist: Optional[utils.DistributionLike] = None,
            snr_dist: Optional[utils.DistributionLike] = None,
            max_overlap: Optional[utils.Numeric] = MAX_OVERLAPPING_EVENTS,
    ):
        # Set attributes passed in by the user
        self.duration = utils.sanitise_positive_number(duration)
        self.fg_path = fg_path
        self.ref_db = ref_db
        # Time overlaps (we could include a space overlaps parameter too)
        self.max_overlap = utils.sanitise_positive_number(max_overlap)

        # Instantiate the `WorldState` object, which loads the mesh and sets up the ray-tracing engine
        if state_kwargs is None:
            state_kwargs = {}
        utils.validate_kwargs(WorldState.__init__, **state_kwargs)
        self.state = WorldState(mesh_path, **state_kwargs)

        # Grab some attributes from the WorldState to make them easier to access
        self.mesh = self.state.mesh
        # self.irs = self.state.irs

        # Add the microphones to the state: note that we will add sources inside `self.add_event`
        self._add_mic_arrays_to_state(mic_arrays)

        # Distributions: these function sanitise the distributions so that they are either `None` or an object
        #  with the `rvs` method. When called, the `rvs` method will return a random variate sampled from the
        #  probability distribution.
        # TODO: these all need to be checked
        self.scene_start_dist = stats.uniform(0., self.duration)
        self.event_start_dist = utils.sanitise_distribution(event_start_dist)
        self.event_duration_dist = utils.sanitise_distribution(event_duration_dist)
        self.event_velocity_dist = utils.sanitise_distribution(event_velocity_dist)
        self.event_resolution_dist = utils.sanitise_distribution(event_resolution_dist)
        self.snr_dist = utils.sanitise_distribution(snr_dist)

        # Assuming path structure with audio files organized in directories per category of interest
        self.fg_category_paths = utils.list_deepest_directories(self.fg_path) if self.fg_path is not None else None

        self.events = {}
        self.ambience_enabled = False

    def _add_mic_arrays_to_state(self, mic_arrays: Any) -> None:
        """
        Populate `self.state.microphones` list with `MicArray` objects
        """
        # Coerce non-iterable types to an iterable
        if isinstance(mic_arrays, (dict, str)) or issubclass(type(mic_arrays), MicArray):
            mic_arrays = [mic_arrays]
        elif not isinstance(mic_arrays, list):
            raise TypeError(f"Cannot handle type {type(mic_arrays)} when adding microphones to space!")
        # Iterate over all the individual microphones we want to place
        for individual_mic in mic_arrays:
            # If a dictionary, try and get all the keyword arguments and use defaults if not present
            if isinstance(individual_mic, dict):
                self.state.add_microphone(
                    microphone_type=individual_mic.get("microphone_type", None),    # defaults to mono capsule
                    position=individual_mic.get("position", None),    # defaults to random position
                    alias=individual_mic.get("alias", None),    # defaults to "mic00N" alias
                )
            # Otherwise, we assume that the object relates to the type of the microphone
            elif isinstance(individual_mic, str) or issubclass(individual_mic, MicArray):
                self.state.add_microphone(microphone_type=individual_mic)
            # Otherwise, we don't know how to parse the input, so raise an error
            else:
                raise TypeError(f"Cannot handle type {type(individual_mic)} when adding microphones to scene")

    def add_ambience(self):
        """Add default room ambience (e.g., Brownian noise)."""
        self.ambience_enabled = True

    def _try_add_event(self, **event_kwargs) -> bool:
        """
        Tries to add an Event with given kwargs.

        Returns:
            bool: True if successful, False otherwise.
        """
        # Grab the alias: this should always be present inside the dictionary
        alias = event_kwargs["alias"]

        # Use only 1 placement attempt if all overrides are present
        has_overrides = all(k in event_kwargs for k in ("scene_start", "event_start", "duration"))
        max_place_attempts = utils.MAX_PLACE_ATTEMPTS if not has_overrides else 1

        # Get emitters from internal state
        emitters = self.state.emitters[alias]

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

            # Sample values (with fallback to override if provided)
            current_kws.update({
                "scene_start": utils.sample_distribution(self.scene_start_dist, overrides["scene_start"]),
                "event_start": utils.sample_distribution(self.event_start_dist, overrides["event_start"]),
                "duration": utils.sample_distribution(self.event_duration_dist, overrides["duration"]),
                "snr": utils.sample_distribution(self.snr_dist, overrides["snr"]),
                "spatial_velocity": utils.sample_distribution(self.event_velocity_dist, overrides["spatial_velocity"]),
                "spatial_resolution": utils.sample_distribution(
                    self.event_resolution_dist, overrides["spatial_resolution"]
                ),
            })

            # Reject this attempt if overlap would be exceeded
            if self._would_exceed_temporal_overlap(current_kws["scene_start"], current_kws["duration"]):
                continue

            # Attempt to create and store the event
            self.events[alias] = Event(**current_kws, emitters=emitters)
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
            for i in os.listdir(fg_category_path):
                if i.endswith(utils.AUDIO_EXTS):
                    audios.append(fg_category_path / Path(i))
        if len(audios) == 0:
            raise ValueError("No audio files found!")
        return utils.sanitise_filepath(random.choice(audios))


    def add_event(
            self,
            filepath: Optional[Union[str, Path]] = None,
            alias: Optional[str] = None,
            emitter_kwargs: Optional[dict] = None,
            event_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Add a foreground event with optional overrides.

        Arguments:
            filepath: a path to a foreground event to use. If not provided, a foreground event will be sampled from
                `fg_category_paths`, if this is provided inside `__init__`; otherwise, an error will be raised.
            alias: the string alias used to index this event inside the `events` dictionary
            emitter_kwargs: a dictionary of keyword arguments that will be passed to `WorldState.add_emitter`.
                These arguments relate to the positionality of the Event within the mesh and its relation to other
                objects within the WorldState. For more information, see `WorldState.add_emitter`.
                If this dictionary is not passed, a single, static Emitter will be added to a random location.
            event_kwargs: a dictionary of keyword arguments to pass to `Event.__init__`.
                These arguments OVERRIDE the probability distributions set inside `Scene.__init__`. In other words,
                passing e.g. `event_start=5.0` into this dictionary will ensure that the Event audio begins at 5
                seconds, without sampling a value from `event_start_dist`. For more information, see `Event.__init__`.

        Examples:
            Creating an event with a predefined position
            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     filepath=...,
            ...     alias="tester",
            ...     emitter_kwargs=dict(
            ...         position=[-0.5, -0.5, 0.5],
            ...         polar=False,
            ...         ensure_direct_path=False,
            ...     )
            ... )

            Creating an event with overrides
            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     filepath=...,
            ...     alias="tester",
            ...     override_kwargs=dict(
            ...         event_start=5.0,
            ...         duration=5.0,
            ...         snr=0.0,
            ...     )
        """

        # Get the alias we'll be using to refer to this event by
        alias = utils.get_default_alias("event", self.events) if alias is None else alias

        # If we haven't provided a filepath, try and sample one from the foreground audio path
        if filepath is None:
            filepath = self._get_random_foreground_audio()

        # Create empty dictionaries if we haven't explicitly provided kwargs
        if emitter_kwargs is None:
            emitter_kwargs = {}
        if event_kwargs is None:
            event_kwargs = {}

        if "sample_rate" in event_kwargs.keys() and event_kwargs["sample_rate"] != self.state.ctx.config.sample_rate:
            raise ValueError("Event sample rate must be the same as the WorldState sample rate")

        # Ensure that we use the same alias for all emitters and events
        emitter_kwargs["alias"] = alias
        event_kwargs["alias"] = alias

        # Add the filepath into the event kwarg dictionary
        event_kwargs["filepath"] = filepath

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
            raise ValueError(f"Could not place event in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                             f"Consider increasing the value of `max_overlap`.")

    def _would_exceed_temporal_overlap(self, new_event_start: float, new_event_duration: float) -> bool:
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

    def generate(self, audio_path, metadata_path, spatial_audio_format='A'):
        """Render scene to disk."""
        audio = render_scene_audio(
            mesh=self.mesh,
            mic_array=self.mic_array,
            events=self.events,
            duration=self.duration,
            ambience=self.ambience_enabled,
            ref_db=self.ref_db,
            spatial_format=spatial_audio_format
        )
        sf.write(audio_path, audio, samplerate=48000) # we shouldn't hard-code this

        metadata = {
            'duration': self.duration,
            'mesh': str(self.mesh),
            'mic_array': self.mic_array.to_dict(),
            'events': [e.to_dict() for e in self.events],
            'ref_db': self.ref_db,
            'ambience': self.ambience_enabled,
            'spatial_format': spatial_audio_format
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def __getitem__(self, alias: str) -> Event:
        return self.get_event(alias)

    def get_event(self, alias: str) -> Event:
        """
        Given a valid alias, get an associated event, as in `self.events[alias]`
        """
        if alias in self.events.keys():
            return self.events[alias]
        else:
            raise KeyError("Emitter alias '{}' not found.".format(alias))

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

    def get_microphone(self, alias: str) -> Type['MicArray']:
        """
        Alias for `WorldState.get_microphone`
        """
        return self.state.get_microphone(alias)

    # noinspection PyProtectedMember
    def clear_events(self) -> None:
        """
        Removes all current events and emitters
        """
        self.events = {}
        self.state._clear_emitters()
