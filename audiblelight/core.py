#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core modules and functions for generation and synthesis."""

import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Union, Optional, Type, Any

import soundfile as sf
from scipy import stats

from audiblelight.event import Event
from audiblelight.micarrays import MicArray
from audiblelight.worldstate import WorldState, Emitter
from audiblelight import utils


MAX_OVERLAPPING_EVENTS = 3
REF_DB = -50


class Scene:
    def __init__(
            self,
            duration: utils.Numeric,
            mesh_path: Union[str, Path],
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

        self.events = OrderedDict()
        self.ambience_enabled = False

        self.audio = None

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
        utils.validate_kwargs(self.state.add_emitter, **kwargs)
        self.state.add_emitter(**kwargs)

    def add_emitters(self, **kwargs):
        """
        Add emitters to the WorldState.

        An alias for `WorldState.add_emitters`: see that method for a full description.
        """
        utils.validate_kwargs(self.state.add_emitters, **kwargs)
        self.state.add_emitters(**kwargs)

    def add_ambience(self):
        """Add default room ambience (e.g., Brownian noise)."""
        self.ambience_enabled = True
        # TODO: implement this

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
            raise FileNotFoundError("No audio files found!")
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
            ...     event_kwargs=dict(
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
        emitter_kwargs["alias"] = alias    # TODO: this will be a problem when we have moving events (multiple emitters)
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

    def generate(
            self,
            audio_path: Union[str, Path] = None,
            metadata_path: Union[str, Path] = None,
            spatial_audio_format: str = "A"
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
            render_scene_audio,
            generate_scene_audio_from_events,
            validate_scene
        )

        # Simulate the IRs for the state
        self.state.simulate()

        # Render all the audio
        #  This populates the `.spatial_audio` attribute inside each Event
        #  It also populates the `audio` attribute inside this instance
        validate_scene(self)
        render_scene_audio(self)
        generate_scene_audio_from_events(self)

        # Write the audio output
        sf.write(audio_path, self.audio.T, int(self.state.ctx.config.sample_rate))

        # Get the metadata and add the spatial audio format in
        metadata = self.to_dict()
        metadata["spatial_format"] = spatial_audio_format

        # Dump the metadata to a JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def __getitem__(self, alias: str) -> Event:
        return self.get_event(alias)

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        # TODO: we should probably add e.g. time, version attributes here: see how MIDITok handles this, it's good
        return dict(
            duration=self.duration,
            ref_db=self.ref_db,
            ambience=self.ambience_enabled,
            events={k: e.to_dict() for k, e in self.events.items()},
            state=self.state.to_dict(),
        )

    def get_event(self, alias: str) -> Event:
        """
        Given a valid alias, get an associated event, as in `self.events[alias]`
        """
        if alias in self.events.keys():
            return self.events[alias]
        else:
            raise KeyError("Event alias '{}' not found.".format(alias))

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
            del self.state.emitters[alias]
            self.state._update()
        else:
            raise KeyError("Event alias '{}' not found.".format(alias))

    def clear_emitters(self) -> None:
        """
        Alias for `WorldState.clear_emitters`.
        """
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
        self.state.clear_emitter(alias)

    def clear_microphone(self, alias: str) -> None:
        """
        Alias for `WorldState.clear_microphone`.
        """
        self.state.clear_microphone(alias)


if __name__ == "__main__":
    from audiblelight.synthesize import render_scene_audio

    sc = Scene(
        duration=30,
        mesh_path=utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb",
        # Pass some default distributions for everything
        event_start_dist=stats.uniform(0, 10),
        event_duration_dist=stats.uniform(0, 10),
        event_velocity_dist=stats.uniform(0, 10),
        event_resolution_dist=stats.uniform(0, 10),
        snr_dist=stats.norm(5, 1),
        fg_path=utils.get_project_root() / "tests/test_resources/soundevents",
        max_overlap=3
    )
    sc.add_microphone(microphone_type="ambeovr")

    for i in range(9):
        sc.add_event(emitter_kwargs=dict(keep_existing=True))
    sc.generate(audio_path="audio_out.wav", metadata_path="metadata_out.json")
