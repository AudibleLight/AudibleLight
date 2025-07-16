#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core modules and functions for generation and synthesis."""

import json
from pathlib import Path
from typing import Union, Optional, Type, Any

import soundfile as sf
import numpy as np

from audiblelight.event import Event
from audiblelight.worldstate import WorldState
from audiblelight.micarrays import MicArray
from audiblelight import utils


def sample_distribution(distribution: Union[utils.DistributionLike, None], override: Union[int, float, None]) -> float:
    """
    Samples from a probability distribution or returns a provided override
    """
    if distribution is None and override is None:
        raise ValueError("Must provide either a probability distribution to sample from or an override")
    elif override is None:
        return distribution.rvs()
    else:
        if isinstance(override, (float, int, complex)):
            return override
        else:
            raise TypeError(f"Expected a numeric input for `override` but got {type(override)}")


class Scene:
    def __init__(
            self,
            duration: Union[int, float],
            mesh_path: Union[str, Path],
            mic_arrays: Union[dict, list[dict], str, list[str], Type['MicArray'], list[Type['MicArray']]],
            fg_path: Union[str, Path],
            state_kwargs: Optional[dict] = None,
            ref_db: Optional[Union[int, float]] = -50,
            event_time_dist: Optional[utils.DistributionLike] = None,
            event_duration_dist: Optional[utils.DistributionLike] = None,
            event_velocity_dist: Optional[utils.DistributionLike] = None,
            snr_dist: Optional[utils.DistributionLike] = None,
            max_overlap: Optional[int] = 3,
    ):
        # Instantiate the `WorldState` object, which loads the mesh and sets up the ray-tracing engine
        if state_kwargs is None:
            state_kwargs = {}
        self.state = WorldState(mesh_path, **state_kwargs)
        self.mesh = self.state.mesh    # grab the mesh from here
        # Add the microphones to the state: note that we will add sources inside `self.add_event`
        self._add_mic_arrays_to_state(mic_arrays)

        self.duration = duration
        self.fg_path = fg_path
        self.ref_db = ref_db

        # Distributions: these function sanitise the distributions so that they are either `None` or an object
        #  with the `rvs` method. When called, the `rvs` method will return a random variate sampled from the
        #  probability distribution.
        self.event_time_dist = utils.sanitise_distribution(event_time_dist)
        self.event_duration_dist = utils.sanitise_distribution(event_duration_dist)
        self.event_velocity_dist = utils.sanitise_distribution(event_velocity_dist)
        self.snr_dist = utils.sanitise_distribution(snr_dist)

        self.max_overlap = max_overlap # time overlaps (we could include a space overlaps parameter too)

        self.events = {}
        self.ambience_enabled = False

    def _add_mic_arrays_to_state(self, mic_arrays: Any) -> None:
        """
        Populate `self.state.microphones` list with `MicArray` objects
        """
        # Coerce non-iterable types to an iterable
        if isinstance(mic_arrays, (dict, str)) or issubclass(mic_arrays, MicArray):
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

    def _try_add_event(
            self,
            time_override: Optional,
            duration_override: Optional,
            snr_override: Optional,
            velocity_override: Optional,
            alias: str,
            filepath: Union[str, Path],
    ) -> bool:
        """
        Tries to add an Event with a specific alias.
        Return True if successful, False otherwise.
        """
        # If we're using both time and duration overrides, only attempt to create the event once
        max_place_attempts = utils.MAX_PLACE_ATTEMPTS if time_override is None or duration_override is None else 1
        # Get the source positions from our state according to our alias
        source = self.state.emitters[alias]
        for attempt in range(max_place_attempts):
            # Sample from distribution or use override values
            event_time = sample_distribution(self.event_time_dist, time_override, )
            event_duration = sample_distribution(self.event_duration_dist, duration_override, )
            # Skip over in cases where the currently sampled time + duration will cause too many overlaps
            if self._would_exceed_overlap(event_time, event_duration):
                continue
            # Adding the event won't exceed the overlap
            #  So, we can proceed to sample SNR and velocity
            snr = sample_distribution(self.snr_dist, snr_override)
            # The event velocity is an interesting case, as it will need to be dependent on:
            #   whether the event is moving or not, its duration
            #   (note possible final location is itself a function of this)
            event_velocity = sample_distribution(self.event_velocity_dist, velocity_override, )
            # Create the event and add to the dictionary
            event = Event(
                filepath=filepath,
                start_timepoint=event_time,
                duration=event_duration,
                snr=snr,
                spatial_velocity=event_velocity,
                start_coordinates_absolute=source,
                sample_rate=self.state.ctx.config.sample_rate,
                # TODO: implement other arguments here
            )
            self.events[alias] = event
            return True
        return False

    def add_event(
            self,
            filepath: Optional[Union[str, Path]] = None,
            # TODO: what does this parameter do?
            source_time: Optional[Union[int, float]] = None,
            event_time: Optional[Union[int, float]] = None,
            duration: Optional[Union[int, float]] = None,
            velocity: Optional[Union[int, float]] = None,
            position: Optional[dict, np.ndarray, list] = None,
            alias: Optional[str] = None,
            snr: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Add a foreground event with optional per-event overrides.

        Examples:
            Creating an event with a predefined position
            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     filepath=...,
            ...     source_time=...,
            ...     event_time=...,
            ...     duration=...,
            ...     velocity=...,
            ...     snr=...,
            ...     alias="...",
            ...     position=dict(
            ...         position=[...],
            ...         polar=True,
            ...         mic_alias=...,
            ...         ensure_direct_path=False,
            ...     )
            ... )

            This is also valid:
            >>> scene = Scene(...)
            >>> scene.add_event(
            ...     filepath=...,
            ...     position=[...]
            ... )

        """
        # Add the source to the mesh, which does the spatial validation
        #  Assume that, if we've just passed in a list, we mean the coordinates in absolute form
        if isinstance(position, (np.ndarray, list)):
            position = dict(position=position)

        # Get the alias we'll be using to refer to
        alias = utils.get_default_alias("event", self.events) if alias is None else alias
        self.state.add_emitter(
            position=position.get("position", None),
            alias=alias,
            mic=position.get("mic_alias", None),
            keep_existing=position.get("keep_existing", True),
            polar=position.get("polar", False),
            ensure_direct_path=position.get("ensure_direct_path", False),
        )

        # Try and create the event: returns True if placed, False if not
        placed = self._try_add_event(event_time, duration, snr, velocity, alias, filepath)

        # Raise an error if we can't place the event correctly
        if not placed:
            raise ValueError(f"Could not place event in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                             f"Consider increasing the value of `max_overlap`.")

    def _would_exceed_overlap(self, new_event_start: float, new_event_duration: float) -> bool:
        """
        Determine whether an event is overlapping with other events more than `max_overlap` times.
        """
        intersections = 0
        new_event_end = new_event_start + new_event_duration
        for event in self.events:
            # Check if intervals [new_start, new_end] and [existing_start, existing_end] overlap
            if new_event_start < event.end_timepoint and new_event_end > event.start_timepoint:
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
