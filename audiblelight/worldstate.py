#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for representing triangular meshes, handling spatial operations, generating RIRs."""

import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from deepdiff import DeepDiff
from loguru import logger
from rlr_audio_propagation import ChannelLayout, ChannelLayoutType, Config, Context

from audiblelight import utils
from audiblelight.micarrays import MICARRAY_LIST, MicArray, sanitize_microphone_input

FACE_FILL_COLOR = [255, 0, 0, 255]

MIN_AVG_RAY_LENGTH = 3.0

EMPTY_SPACE_AROUND_EMITTER = 0.2  # Minimum distance one emitter can be from another
EMPTY_SPACE_AROUND_MIC = 0.1  # Minimum distance one emitter can be from the mic
EMPTY_SPACE_AROUND_SURFACE = 0.2  # Minimum distance from the nearest mesh surface
EMPTY_SPACE_AROUND_CAPSULE = (
    0.05  # Minimum distance from individual microphone capsules
)

WARN_WHEN_EFFICIENCY_BELOW = (
    0.5  # when the ray efficiency is below this value, raise a warning in .simulate
)

MOVING_EMITTER_MAX_SPEED = 1  # meters per second
MOVING_EMITTER_TEMPORAL_RESOLUTION = (
    4  # number of IRs created per second for a moving emitter
)


def load_mesh(mesh_fpath: Union[str, Path]) -> trimesh.Trimesh:
    """
    Loads a mesh from disk and coerces units to meters
    """
    # Load up in trimesh, setting the metadata dictionary nicely
    #  This just allows us to access the filename, etc., later
    mesh_fpath = utils.sanitise_filepath(mesh_fpath)
    metadata = dict(
        fname=mesh_fpath.stem, ftype=mesh_fpath.suffix, fpath=str(mesh_fpath)
    )
    # noinspection PyTypeChecker
    loaded_mesh = trimesh.load_mesh(
        mesh_fpath, file_type=mesh_fpath.suffix, metadata=metadata
    )
    # Convert the units of the mesh to meters, if this is not provided
    if loaded_mesh.units != utils.MESH_UNITS:
        logger.warning(
            f"Mesh {mesh_fpath.stem} has units {loaded_mesh.units}, converting to {utils.MESH_UNITS}"
        )
        loaded_mesh = loaded_mesh.convert_units(utils.MESH_UNITS, guess=True)
    return loaded_mesh


def get_broken_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Get the idxs of broken faces in a mesh. Uses copies to prevent anything being set inplace."""
    # Make a copy of the mesh
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Get the idxs of the faces in the mesh which break the watertight status of the mesh.
    return trimesh.repair.broken_faces(new_mesh, color=FACE_FILL_COLOR)


def repair_mesh(mesh: trimesh.Trimesh) -> None:
    """
    Uses Trimesh functionality to repair a mesh when necessary
    """
    # These functions all operate inplace
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)
    # Now see how many faces are broken after repairing: also, fill in their face color
    broken_faces_new = trimesh.repair.broken_faces(mesh, color=FACE_FILL_COLOR)
    logger.info(f"Broken faces after repair: {len(broken_faces_new)}")


def add_sphere(
    scene: trimesh.Scene, pos: np.array, color: list[int] = None, r: float = 0.2
) -> None:
    """Adds a sphere object to a scene with given position, color, and radius"""
    if color is None:
        color = [0, 0, 0]
    sphere = trimesh.creation.uv_sphere(radius=r)
    sphere.apply_translation(pos)
    sphere.visual.face_colors = color
    scene.add_geometry(sphere)


class Emitter:
    """
    Represents an *individual* position for a sound source within a mesh.

    The `Emitter` object handles all information with respect to a single sound source at a single position. This
    includes its absolute coordinates within the mesh, as well as its relative position (Cartesian + polar) compared
    to all other `MicArray` and `Emitter` instances.

    Note that, in the case of a static (non-moving) audio source, a single `Event` will be associated with a single
    `Emitter`. In the case of a *moving* audio source, we will instead have multiple `Emitter` objects per `Event`.
    """

    def __init__(self, alias: str, coordinates_absolute: np.ndarray):
        self.alias: str = alias
        self.coordinates_absolute: np.ndarray = utils.sanitise_coordinates(
            coordinates_absolute
        )
        # These dictionaries map from {alias: position} for all other emitter and microphone array objects
        self.coordinates_relative_cartesian: Optional[OrderedDict[str, np.ndarray]] = (
            OrderedDict()
        )
        self.coordinates_relative_polar: Optional[OrderedDict[str, np.ndarray]] = (
            OrderedDict()
        )

    # noinspection PyUnresolvedReferences
    def update_coordinates(
        self,
        coordinates: OrderedDict[str, Union[Type["MicArray"], list[Type["Emitter"]]]],
    ):
        """
        Updates coordinates of this emitter WRT a dictionary in the format {alias: MicArray | list[Emitter]}
        """
        for alias, obj in coordinates.items():
            # Add zero-arrays if the object is the current Emitter
            # TODO: note that this won't currently work for moving emitters
            if alias == self.alias:
                self.coordinates_relative_cartesian[alias] = np.array([0.0, 0.0, 0.0])
                self.coordinates_relative_polar[alias] = np.array([0.0, 0.0, 0.0])

            else:
                # Grab the coordinates from the object: these should all be in Cartesian, XYZ format
                #  For micarrays, use the center of all capsules; for emitters, use the absolute position
                if issubclass(type(obj), MicArray):
                    coords = utils.sanitise_coordinates(obj.coordinates_center)

                elif isinstance(obj, list):
                    assert all([isinstance(em, Emitter) for em in obj])
                    # TODO: check that this makes sense
                    coords = np.vstack([em.coordinates_absolute for em in obj])

                else:
                    raise TypeError(
                        "Cannot handle input with type {}".format(type(obj))
                    )

                # Express the position of the CURRENT emitter WRT the object we're considering
                pos = self.coordinates_absolute - coords
                self.coordinates_relative_cartesian[alias] = pos
                self.coordinates_relative_polar[alias] = utils.cartesian_to_polar(pos)

    def __repr__(self) -> str:
        """
        Returns a JSON-formatted string representation of the Emitter
        """
        return utils.repr_as_json(self)

    def __str__(self) -> str:
        """
        Returns a string representation of the Emitter
        """
        return (
            f"Emitter '{self.alias}' with absolute position {self.coordinates_absolute}"
        )

    def __eq__(self, other: Any):
        """
        Compare two Emitter objects for equality.

        Returns:
            bool: True if two Emitter objects are equal, False otherwise
        """

        # Non-Emitter objects are always not equal
        if not isinstance(other, Emitter):
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
        Returns a dictionary representation of the Emitter

        Returns:
            dict
        """

        def coerce(inp: Any) -> Any:
            """Coerce dtypes for JSON serialisation"""
            if isinstance(inp, dict):
                return {k: coerce(v) for k, v in inp.items()} if inp else None
            elif isinstance(inp, np.ndarray):
                return inp.tolist()
            else:
                return inp

        return dict(
            alias=self.alias,
            coordinates_absolute=coerce(self.coordinates_absolute),
            coordinates_relative_cartesian=coerce(self.coordinates_relative_cartesian),
            coordinates_relative_polar=coerce(self.coordinates_relative_polar),
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate an `Emitter` from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `Emitter`.

        Returns:
            Emitter instance.
        """
        # Don't modify the input dictionary
        copied_dict = deepcopy(input_dict)

        def unserialise(inp: Any) -> Any:
            if isinstance(inp, dict):
                return {k_: unserialise(v) for k_, v in inp.items()} if inp else None
            elif isinstance(inp, list):
                return np.asarray(inp)
            else:
                return inp

        # Sanity check the keys are correct
        for k in [
            "alias",
            "coordinates_absolute",
            "coordinates_relative_cartesian",
            "coordinates_relative_polar",
        ]:
            if k not in copied_dict:
                raise KeyError(f"Missing key '{k}'")

            # Need to convert lists back to arrays
            copied_dict[k] = unserialise(copied_dict[k])

        # Instantiate the class with the correct alias and absolute coordinates
        instantiated = cls(
            alias=copied_dict["alias"],
            coordinates_absolute=copied_dict["coordinates_absolute"],
        )

        # Set the relative coordinates correctly
        setattr(
            instantiated,
            "coordinates_relative_cartesian",
            copied_dict["coordinates_relative_cartesian"],
        )
        setattr(
            instantiated,
            "coordinates_relative_polar",
            copied_dict["coordinates_relative_polar"],
        )

        return instantiated


class WorldState:
    """
    Represents a 3D space defined by a mesh, microphone position(s), and emitter position(s)

    This class is capable of handling spatial operations and simulating audio propagation using the ray-tracing library.

    Attributes:
        mesh (str, Path): The path to the mesh on the disk.
        microphones (np.array): Position of the microphone in the mesh.
        ctx (rlr_audio_propagation.Context): The context for audio propagation simulation.
        emitters (np.array): relative positions of sound emitter

    """

    def __init__(
        self,
        mesh: Union[str, Path],
        empty_space_around_mic: Optional[float] = EMPTY_SPACE_AROUND_MIC,
        empty_space_around_emitter: Optional[float] = EMPTY_SPACE_AROUND_EMITTER,
        empty_space_around_surface: Optional[float] = EMPTY_SPACE_AROUND_SURFACE,
        empty_space_around_capsule: Optional[float] = EMPTY_SPACE_AROUND_CAPSULE,
        repair_threshold: Optional[float] = None,
        rlr_kwargs: Optional[dict] = None,
    ):
        """
        Initializes the WorldState with a mesh and sets up the audio context.

        Args:
            mesh (str|Path): The name of the mesh file. Units will be coerced to meters when loading
            empty_space_around_mic (float): minimum meters new emitters/mics will be placed from center of other mics
            empty_space_around_emitter (float): minimum meters new emitters/mics will be placed from other emitters
            empty_space_around_surface (float): minimum meters new emitters/mics will be placed from mesh emitters
            empty_space_around_capsule (float): minimum meters new emitters/mics will be placed from mic capsules
            repair_threshold (float, optional): when the proportion of broken faces on the mesh is below this value,
                repair the mesh and fill holes. If None, will never repair the mesh.
            rlr_kwargs (dict, optional): additional keyword arguments to pass to the RLR audio propagation library.
                For instance, sample rate can be set by passing `rlr_kwargs=dict(sample_rate=...)`
        """
        # Store emitter and mic positions in here to access later; these should be in ABSOLUTE form
        self.emitters = OrderedDict()
        self.microphones = OrderedDict()
        self._irs = None  # will be updated when calling `simulate`

        # Distances from objects/mesh surfaces
        self.empty_space_around_mic = empty_space_around_mic
        self.empty_space_around_surface = empty_space_around_surface
        self.empty_space_around_emitter = empty_space_around_emitter
        self.empty_space_around_capsule = empty_space_around_capsule

        # Load in the trimesh object
        self.mesh = load_mesh(mesh)

        # If we want to try and repair the mesh, and if it actually needs repairing
        self.repair_threshold = repair_threshold
        if self.repair_threshold is not None and not self.mesh.is_watertight:
            # Get the idxs of faces in the mesh that break the watertight status
            broken_faces = get_broken_faces(
                self.mesh
            )  # this uses copies so nothing will be set in-place
            # If the proportion of broken faces is below the desired threshold, do the repair in-place
            if len(broken_faces) / self.mesh.faces.shape[0] < repair_threshold:
                repair_mesh(self.mesh)

        # Setting up audio context
        cfg = self._parse_rlr_config(rlr_kwargs)
        self.ctx = Context(cfg)
        self._setup_audio_context()

    def _update(self) -> None:
        """
        Updates the state, setting emitter positions and adding all items to the ray-tracing context correctly.
        """
        # Update the ray-tracing listeners
        self.ctx.clear_listeners()
        if len(self.microphones) > 0:
            all_caps = np.vstack(
                [m.coordinates_absolute for m in self.microphones.values()]
            )
            for caps_idx, caps_pos in enumerate(all_caps):  # type: np.ndarray
                # Add a single listener for each individual capsule
                self.ctx.add_listener(ChannelLayout(ChannelLayoutType.Mono, 1))
                self.ctx.set_listener_position(caps_idx, caps_pos.tolist())

        # Update the ray-tracing sources
        #  We have to clear sources out regardless of number of emitters because it is possible that, if we have
        #  removed an event (e.g. `Scene.remove_event(...)`), we'll "orphan" some sources otherwise
        self.ctx.clear_sources()
        if len(self.emitters) > 0:
            emitter_counter = 0
            for emitter_alias, emitter_list in self.emitters.items():
                for emitter in emitter_list:
                    # Update the coordinates of the emitter WRT other microphones, emitters
                    emitter.update_coordinates(self.emitters)
                    emitter.update_coordinates(self.microphones)
                    # Add the emitter to the ray-tracing engine
                    self.ctx.add_source()
                    pos = emitter.coordinates_absolute
                    self.ctx.set_source_position(
                        emitter_counter,
                        pos.tolist() if isinstance(pos, np.ndarray) else pos,
                    )
                    # Update the counter used in the ray-tracing engine by one
                    emitter_counter += 1

    @staticmethod
    def _parse_rlr_config(rlr_kwargs: dict) -> Config:
        """
        Parses the configuration for the ray-tracing engine
        """
        # Create the configuration object with the default settings
        cfg = Config()
        if rlr_kwargs is None:
            rlr_kwargs = {}
        # Iterate over our passed parameters and update as required
        for rlr_kwarg, rlr_val in rlr_kwargs.items():
            if hasattr(cfg, rlr_kwarg):
                setattr(cfg, rlr_kwarg, rlr_val)
            else:
                raise AttributeError(f"Ray-tracing engine has no attribute {rlr_kwarg}")
        return cfg

    @property
    def irs(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of IRs in the shape {mic000: (N_capsules, N_emitters, N_samples), mic001: (...)}
        """
        if self._irs is None:
            raise AttributeError(
                "IRs have not been simulated yet: add microphones and emitters and call `simulate`."
            )
        else:
            return self._irs

    def calculate_weighted_average_ray_length(
        self, point: np.ndarray, num_rays: int = 100
    ) -> float:
        """
        Calculate the weighted average length of rays cast from a point into a mesh.
        """
        # Generate random azimuthal angles for each ray
        angles = np.random.uniform(0, 2 * np.pi, num_rays)
        # Generate random elevation angles for each ray
        elevations = np.random.uniform(-np.pi / 2, np.pi / 2, num_rays)
        # Convert spherical coordinates (angles, elevations) to  Cartesian 3D direction vectors
        directions = np.column_stack(
            [
                np.cos(elevations) * np.cos(angles),  # X component
                np.cos(elevations) * np.sin(angles),  # Y component
                np.sin(elevations),  # Z component
            ]
        )
        # Repeat the origin point for each ray so rays start from the same position
        origins = np.tile(point, (num_rays, 1))
        # Cast rays from the origin in the computed directions and find the longest intersection distances with the mesh
        distances = trimesh.proximity.longest_ray(self.mesh, origins, directions)
        # We can get `inf` values here, likely due to holes in the mesh causing a ray never to intersect
        if any(np.isinf(distances)):
            # For simplicity, we can just remove these here but raise a warning
            logger.warning(
                f"Some rays cast from point {point} have infinite distances: is the mesh watertight?"
            )
            distances = distances[distances != np.inf]
        # Compute weights by squaring the distances to give more importance to longer rays
        weights = distances**2
        # Calculate weighted average of the distances using the computed weights
        weighted_average = np.sum(distances * weights) / np.sum(weights)
        # Return the weighted average ray length
        return weighted_average

    def _setup_audio_context(self) -> None:
        """
        Initializes the audio context and configures the mesh for the context.
        """
        self.ctx.add_object()
        self.ctx.add_mesh_vertices(self.mesh.vertices.flatten().tolist())
        self.ctx.add_mesh_indices(self.mesh.faces.flatten().tolist(), 3, "default")
        self.ctx.finalize_object_mesh(0)

    def _try_add_microphone(
        self, mic_cls, position: Union[list, None], alias: str
    ) -> bool:
        """
        Try to place a microphone of type mic_cls at position with given alias. Return True if successful.
        """
        if alias in self.microphones.keys():
            raise KeyError(f"Alias {alias} already exists in microphone dictionary")

        for attempt in range(utils.MAX_PLACE_ATTEMPTS):
            # Grab a random position for the microphone if required
            pos = position if position is not None else self.get_random_position()
            assert len(pos) == 3, f"Expected three coordinates but got {len(pos)}"
            # Instantiate the microphone and set its coordinates
            mic = mic_cls()
            mic.set_absolute_coordinates(pos)
            # If we have a valid position for the microphone
            if all(self._validate_position(caps) for caps in mic.coordinates_absolute):
                self.microphones[alias] = mic
                return True
            # If we were trying to place the microphone in a specific location, only make one attempt at placing it
            elif position is not None:
                break
        return False

    @utils.update_state
    def add_microphone(
        self,
        microphone_type: Union[str, Type["MicArray"], None] = None,
        position: Union[list, np.ndarray, None] = None,
        alias: str = None,
        keep_existing: bool = True,
    ) -> None:
        """
        Add a microphone to the space.

        Arguments:
            microphone_type: Type of microphone to add, defaults to a mono capsule.
            position: Location to add the microphone in absolute cartesian units, defaults to a random, valid location.
            alias: String reference to access the microphone inside the `self.microphones` dictionary.
            keep_existing (optional): whether to keep existing microphones from the mesh or remove, defaults to keep

        Examples:
            Create a state from a given mesh
            >>> spa = WorldState(mesh=...)

            Add a AmbeoVR microphone with a random position and default alias
            >>> spa.add_microphone("ambeovr")
            >>> spa.microphones["mic000"]    # access with default alias

            Alternative, using `MicArray` objects
            >>> from audiblelight.micarrays import AmbeoVR
            >>> spa.add_microphone(AmbeoVR)

            Add AmbeoVR with given position and alias
            >>> spa.add_microphone(microphone_type="ambeovr", position=[0.5, 0.5, 0.5], alias="ambeo")
            >>> spa.microphones["ambeo"]    # access using given alias
        """
        # TODO: consider removing
        # Remove existing microphones if we wish to do this
        if not keep_existing:
            self.clear_microphones()

        # Get the correct microphone type.
        sanitized_microphone = sanitize_microphone_input(microphone_type)

        # Get the microphone alias
        alias = (
            utils.get_default_alias("mic", self.microphones) if alias is None else alias
        )

        # Try and place the microphone inside the space
        placed = self._try_add_microphone(sanitized_microphone, position, alias)

        # If we can't add the microphone to the mesh
        if not placed:
            # If we were trying to add it to a random position
            if position is None:
                raise ValueError(
                    f"Could not place microphone in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                    f"Consider reducing `empty_space_around` arguments."
                )
            # If we were trying to add it to a specific position
            else:
                raise ValueError(
                    f"Position {position} invalid for microphone {sanitized_microphone.name}. "
                    f"Consider reducing `empty_space_around` arguments."
                )

    @utils.update_state
    def add_microphones(
        self,
        microphone_types: list[Union[str, Type["MicArray"], None]] = None,
        positions: list[Union[list, np.ndarray, None]] = None,
        aliases: list[str] = None,
        keep_existing: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        """
        Add multiple microphones to the mesh.

        This function essentially takes in lists of the arguments expected by `add_microphone`. The `raise_on_error`
        command will skip over microphones that cannot be placed in the mesh and raise a warning in the console.

        Arguments:
            microphone_types: Types of microphones to add, defaults to a single mono capsule.
            positions: Locations to add the microphones in absolute cartesian units, defaults to a single location.
            aliases: String references to access the microphones inside the `self.microphones` dictionary.
            keep_existing (optional): whether to keep existing microphones from the mesh or remove, defaults to keep
            raise_on_error (optional): if True, will raise an error when unable to place a mic, otherwise skips to next

        Examples:
            Create a state with a given mesh
            >>> spa = WorldState(mesh=...)

            Add some AmbeoVRs with random positions
            >>> spa.add_microphones(microphone_types=["ambeovr", "ambeovr", "ambeovr"])
            >>> spa.microphones["mic002"]    # access with default alias

            Add AmbeoVR and Eigenmike32 with given positions and aliases
            >>> spa.add_microphones(
            >>>     microphone_types=["ambeovr", "eigenmike32"],
            >>>     positions=[[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
            >>>     alias=["ambeo", "eigen"],
            >>>     keep_existing=False,     # removes microphones already added to the space
            >>>     raise_on_error=True,    # raises an error if any microphone cannot be placed
            >>> )
            >>> spa.microphones["eigen"]    # access using given alias
        """

        # Remove existing microphones if we wish to do this
        if not keep_existing:
            self.clear_microphones()

        # Handle cases with non-unique aliases
        if aliases is not None:
            if len(set(aliases)) != len(aliases):
                raise ValueError("Only unique aliases can be passed")

        all_not_none = [
            l_ for l_ in [microphone_types, positions, aliases] if l_ is not None
        ]
        # Handle cases where we haven't provided an equal number of mic types, positions, and aliases
        if not utils.check_all_lens_equal(*all_not_none):
            raise ValueError("Expected all inputs to have equal length")

        # Get the index to iterate up to
        max_idx = max([len(a) for a in all_not_none]) if len(all_not_none) > 0 else 0
        # Iterate over all the microphones we want to place
        for idx in range(max_idx):
            microphone_type_ = (
                microphone_types[idx] if microphone_types is not None else None
            )
            position_ = positions[idx] if positions is not None else None
            alias_ = aliases[idx] if aliases is not None else None

            # Get the correct microphone type.
            sanitized_microphone = sanitize_microphone_input(microphone_type_)

            # Get the microphone alias
            alias_ = (
                utils.get_default_alias("mic", self.microphones)
                if alias_ is None
                else alias_
            )

            # Try and place the microphone inside the space
            placed = self._try_add_microphone(sanitized_microphone, position_, alias_)

            # If we can't add the microphone to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if position_ is None:
                    msg = (
                        f"Could not place microphone in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                        f"Consider reducing `empty_space_around` arguments."
                    )
                # If we were trying to add it to a specific position
                else:
                    msg = (
                        f"Position {position_} invalid for microphone {sanitized_microphone.name}. "
                        f"Consider reducing `empty_space_around` arguments."
                    )

                # Raise the error if required or just log a warning and skip to the next microphone
                if raise_on_error:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

    @utils.update_state
    def add_microphone_and_emitter(
        self,
        position: Optional[Union[np.ndarray, float]] = None,
        polar: Optional[bool] = True,
        microphone_type: Optional[Union[str, Type["MicArray"]]] = None,
        mic_alias: Optional[str] = None,
        emitter_alias: Optional[str] = None,
        keep_existing_mics: Optional[bool] = True,
        keep_existing_emitters: Optional[bool] = True,
        ensure_direct_path: Optional[bool] = True,
        max_place_attempts: Optional[int] = utils.MAX_PLACE_ATTEMPTS,
    ) -> None:
        """
        Add both a microphone and emitter with specified relationship.

        The microphone will be placed in a random, valid position. The emitter will then be placed relative to the
        microphone, either in Cartesian or spherical coordinates.

        Args:
            position (np.ndarray): Array of form [X, Y, Z]
            polar: whether the coordinates are provided in spherical form. If True:
                - Azimuth (X) must be between 0 and 360
                - Colatitude (Y) must be between 0 and 180
                - Elevation (Z) must be a positive value, measured in the same units given by the mesh.
            microphone_type: Type of microphone to add, defaults to mono capsule
            mic_alias: String reference for the microphone, auto-generated if None
            emitter_alias: String reference for the emitter, auto-generated if None
            keep_existing_mics: Whether to keep existing microphones, defaults to True
            keep_existing_emitters: Whether to keep existing emitters, defaults to True
            ensure_direct_path: Whether to ensure line-of-sight between mic and emitter
            max_place_attempts: The number of times to try placing the microphone and emitter

        Raises:
            ValueError: If unable to place microphone and emitter within the mesh

        Examples:
            # Create a state with a given mesh
            >>> spa = WorldState(mesh=...)

            # Place emitter 2 meters in front of microphone
            >>> spa.add_microphone_and_emitter(np.array([0, 0, 2.0]))

            # Place emitter 1.5 meters to the left and slightly above
            >>> spa.add_microphone_and_emitter(np.array([90, 30, 1.5]), mic_alias="main_mic", emitter_alias="left_source")

            # Place emitter behind and below
            >>> spa.add_microphone_and_emitter(np.array([180, -45, 1.0]))
        """

        # Sanitise the input coordinates and microphone type
        emitter_offset = utils.sanitise_coordinates(position)
        sanitized_microphone = sanitize_microphone_input(microphone_type)

        # Remove existing objects if requested
        if not keep_existing_mics:
            self.clear_microphones()
        if not keep_existing_emitters:
            self.clear_emitters()

        # Get aliases
        mic_alias = (
            utils.get_default_alias("mic", self.microphones)
            if mic_alias is None
            else mic_alias
        )
        emitter_alias = (
            utils.get_default_alias("src", self.emitters)
            if emitter_alias is None
            else emitter_alias
        )

        # Convert spherical coordinates to Cartesian offset if required
        if polar:
            emitter_offset = utils.polar_to_cartesian(emitter_offset)[
                0
            ]  # returns a 2D array, we just want 1D

        # Attempt to find valid positions for both microphone and emitter
        for attempt in range(max_place_attempts):
            # Get a random position for the microphone
            mic_pos = self.get_random_position()

            # Calculate emitter position based on spherical coordinates
            emitter_pos = mic_pos + emitter_offset

            # Create temporary microphone to test position validity
            temp_mic = sanitized_microphone()
            temp_mic.set_absolute_coordinates(mic_pos)

            # Validate both positions
            mic_valid = all(
                self._validate_position(caps) for caps in temp_mic.coordinates_absolute
            )
            emitter_valid = self._validate_position(emitter_pos)

            # Check direct path if required
            direct_path_ok = True
            if ensure_direct_path:
                direct_path_ok = self.path_exists_between_points(
                    temp_mic.coordinates_center, emitter_pos
                )

            # If all conditions are met, place both objects
            if mic_valid and emitter_valid and direct_path_ok:
                # Add microphone
                self.microphones[mic_alias] = temp_mic

                # Add emitter
                emitter = Emitter(alias=emitter_alias, coordinates_absolute=emitter_pos)

                # If we already have emitters under this alias, add to the list, otherwise create a new entry
                #  This is so we can have multiple emitters under one alias in the case of moving sound sources
                if emitter_alias in self.emitters:
                    self.emitters[emitter_alias].append(emitter)
                else:
                    self.emitters[emitter_alias] = [emitter]

                logger.info(
                    f"Successfully placed microphone and emitter after {attempt + 1} attempts"
                )
                logger.info(f"Microphone '{mic_alias}' at: {mic_pos}")
                logger.info(f"Emitter '{emitter_alias}' at: {emitter_pos}")
                return

            # Log progress every 100 attempts
            if (attempt + 1) % 100 == 0:
                logger.info(f"Placement attempt {attempt + 1}/{max_place_attempts}")

        # If we reach here, we couldn't place the objects
        raise ValueError(
            f"Could not place microphone and emitter with specified relationship "
            f"after {max_place_attempts} attempts. Consider:\n"
            f"- Reducing the distance between emitter and microphone ({emitter_offset[-1]}m may be too large for the mesh)\n"
            f"- Reducing `empty_space_around parameters`\n"
            f"- Setting `ensure_direct_path=False` if line-of-sight is not required\n"
            f"- Increasing `max_placement_attempts` (currently {max_place_attempts})"
        )

    def get_random_position(self) -> np.ndarray:
        """
        Get a random position to place a emitter inside the mesh
        """
        # Get an initial microphone position
        mic_pos = self.get_random_point_inside_mesh()
        # Start iterating until we get an acceptable position
        for attempt in range(utils.MAX_PLACE_ATTEMPTS):
            # Compute the weighted average ray length with this position
            avg_ray_length = self.calculate_weighted_average_ray_length(mic_pos)
            # If the position is acceptable, break out
            if avg_ray_length >= MIN_AVG_RAY_LENGTH:
                logger.info(f"Found suitable position after {attempt + 1} attempts")
                break
            # Otherwise, try again with a new position
            else:
                mic_pos = self.get_random_point_inside_mesh()
        # If we haven't found an acceptable position, log this and use the most recent one.
        else:
            logger.error(
                f"Could not find a suitable position after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                f"Using the last attempted position, which is {mic_pos}."
            )
        return mic_pos

    def get_random_point_inside_mesh(self) -> np.ndarray:
        """
        Generates a random valid point inside the mesh.

        Returns:
            np.array: A valid point within the mesh in XYZ format
        """
        while True:
            point = np.random.uniform(self.mesh.bounds[0], self.mesh.bounds[1])
            # This checks e.g. distance from surface, other emitters, other mics, and that point is in-bounds
            if self._validate_position(point):
                return point

    def _is_point_inside_mesh(self, point: Union[np.array, list]) -> bool:
        """
        Determines whether a given point is inside the mesh.

        Args:
            point (np.array, list): The point to check.

        Returns:
            bool: True if the point is inside the mesh, otherwise False.
        """
        return bool(self.mesh.contains(utils.coerce2d(point))[0])

    def _validate_position(self, pos_abs: np.ndarray) -> bool:
        """
        Validates a position or array of positions with respect to the mesh and objects inside it.
        Returns True if valid, False if not. If multiple arrays provided, return True only if all are valid.
        """
        # Coerce to a 2D array of XYZ positions, for iteration
        positions = utils.coerce2d(pos_abs)
        if positions.shape[1] != 3:
            raise ValueError("Expected input to have shape (N, 3) for XYZ coordinates")

        # Iterate over all positions
        for position in positions:
            # Check minimum distance from all emitters
            for emitter_list in self.emitters.values():
                for emitter in emitter_list:
                    if (
                        np.linalg.norm(position - emitter.coordinates_absolute)
                        < self.empty_space_around_emitter
                    ):
                        return False

            # Check minimum distance from the center of every microphone and from every individual capsule
            if len(self.microphones) > 0:
                for attr, thresh in zip(
                    # check mic centers first, check mic capsules second
                    ["coordinates_center", "coordinates_absolute"],
                    [self.empty_space_around_mic, self.empty_space_around_capsule],
                ):
                    coordinates = np.vstack(
                        [getattr(mic, attr) for mic in self.microphones.values()]
                    )
                    distances = np.linalg.norm(position - coordinates, axis=1)
                    if np.any(distances < thresh):
                        return False

            # Check minimum distance from mesh surface
            if (
                self.mesh.nearest.on_surface([position])[1][0]
                < self.empty_space_around_surface
            ):
                return False

            # Check if the position is inside the mesh
            if not self._is_point_inside_mesh(position):
                return False

        return True

    def _try_add_emitter(
        self,
        position: Optional[list],
        relative_mic: Optional[Type["MicArray"]],
        alias: str,
        polar: bool,
        path_between: list[str],
    ) -> bool:
        """
        Attempt to add a emitter at the given position with the specified alias.
        Returns True if placement is successful, otherwise False.
        """
        # True if we want a specific position, False if not
        position_is_assigned = position is not None
        # If we have already provided a position, this loop will only iterate once
        #  Otherwise, we want a random position, so we iterate N times until the position is valid
        for attempt in range(1 if position_is_assigned else utils.MAX_PLACE_ATTEMPTS):
            # Get a random position if required or use the assigned one
            pos = position if position_is_assigned else self.get_random_position()
            if len(pos) != 3:
                raise ValueError(f"Expected three coordinates but got {len(pos)}")
            # Convert to Cartesian if position is in polar coordinates
            if polar:
                if not relative_mic or not position_is_assigned:
                    raise ValueError(
                        "Polar coordinates require a relative mic and a fixed position"
                    )
                pos = utils.polar_to_cartesian(pos)[0]
            # Adjust position relative to the mic array if provided
            if relative_mic:
                pos = relative_mic.coordinates_center + pos
            # If position invalid, skip over
            if not self._validate_position(pos):
                continue
            # If line-of-sight not obtained with required microphones, skip over
            if not all(
                self.path_exists_between_points(
                    pos, self.microphones[d].coordinates_center
                )
                for d in path_between
            ):
                continue
            # Successfully placed: add to the emitter dictionary and return True
            #  We will update the `coordinates_relative` objects in the `update_state` decorator
            emitter = Emitter(alias=alias, coordinates_absolute=np.asarray(pos))
            # Add the emitter to the list created for this alias, or create the list if it doesn't exist
            # TODO: we create a list of emitters for both static and moving sound sources
            #  when moving, len(emitters) > 1, when static, len(emitters) == 1
            if alias in self.emitters:
                self.emitters[alias].append(emitter)
            else:
                self.emitters[alias] = [emitter]
            return True
        # Cannot place: return False
        return False

    def _get_mic_from_alias(
        self, mic_alias: Optional[str]
    ) -> Optional[Type["MicArray"]]:
        """Get a given `MicArray` object from its alias"""
        if mic_alias is not None:
            if mic_alias not in self.microphones:
                raise KeyError(f"No microphone found with alias {mic_alias}!")
            return self.microphones[mic_alias]
        else:
            return None

    def path_exists_between_points(
        self, point_a: np.ndarray, point_b: np.ndarray
    ) -> bool:
        """
        Returns True if a direct point exists between point_a and point_b in the mesh, False otherwise.
        """
        # Coerce to 1D array and sanity check
        point_a = np.asarray(point_a)
        point_b = np.asarray(point_b)
        for point in [point_a, point_b]:
            assert point.shape == (
                3,
            ), f"Expected an array with shape (3, ) but got {point.shape}"
            # If a point is not inside the mesh, we shouldn't expect a direct path
            if not self._is_point_inside_mesh(point):
                return False
        # Calculate direction vector from points A to B
        direction = point_b - point_a
        length = np.linalg.norm(direction)
        direction_unit = direction / length
        # Cast ray from A towards B and get intersections (locations and indices)
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=utils.coerce2d(point_a),  # trimesh expecting 2D arrays?
            ray_directions=utils.coerce2d(direction_unit),
        )
        # Check if any intersection is closer than B
        if len(locations) > 0:
            # Calculate distances from A to each intersection
            distances = np.linalg.norm(locations - point_a, axis=1)
            if np.any(distances < length):
                # No direct line: mesh blocks the segment.
                return False
        # Direct line exists: either no blocking intersections, or no intersections at all
        return True

    def _parse_valid_microphone_aliases(
        self, aliases: Union[bool, list, str, None]
    ) -> list[str]:
        """
        Get valid microphone aliases from an input
        """
        # If True, we should get a list of all the microphones
        if aliases is True:
            return list(self.microphones.keys())
        # If a single string, validate and convert to [string]
        elif isinstance(aliases, str):
            if aliases not in self.microphones.keys():
                raise KeyError(f"Alias {aliases} is not a valid microphone alias!")
            return [aliases]
        # If a list of strings, validate these
        elif isinstance(aliases, list):
            # Sanity check that all the provided aliases exist in our dictionary
            not_in = [e for e in aliases if e not in self.microphones.keys()]
            if len(not_in) > 0:
                raise KeyError(
                    f"Some provided microphone aliases were not found: {', '.join(not_in)}"
                )
            # Remove duplicates from the list
            return list(set(aliases))
        # If False or None, return an empty list (which we'll skip over later)
        elif aliases is False or aliases is None:
            return []
        # Otherwise, we can't handle the input, so return an error
        else:
            raise TypeError(f"Cannot handle input with type {type(aliases)}")

    @utils.update_state
    def add_emitter(
        self,
        position: Optional[Union[list, np.ndarray]] = None,
        alias: Optional[str] = None,
        mic: Optional[str] = None,
        keep_existing: Optional[bool] = False,
        polar: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
    ) -> None:
        """
        Add a emitter to the state.

        If `mic` is a key inside `microphones`, `position` is assumed to be relative to that microphone; else,
        it is assumed to be in absolute terms. If `polar` is True, `position` should be in the form
        (azimuth°, polar°, radius); else, it should be in cartesian coordinates in meters with the form [x, y, z].
        Note that `mic_alias` must not be None when `polar` is True.

        Arguments:
            position: Location to add the emitter, defaults to a random, valid location.
            alias: String reference to access the emitter inside the `self.emitters` dictionary.
            mic: String reference to a microphone inside `self.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            keep_existing (optional): Whether to keep existing emitters from the mesh or remove, defaults to keep
            polar: When True, expects `position` to be provided in [azimuth, colatitude, elevation] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.

        Examples:
            Create a state with a given mesh and add a microphone
            >>> spa = WorldState(mesh=...)
            >>> spa.add_microphone(alias="tester")

            Add a single emitter with a random position
            >>> spa.add_emitter()
            >>> spa.get_emitter("src000")    # access with default alias

            Add emitter with given position and alias
            >>> spa.add_emitter(position=[0.5, 0.5, 0.5], alias="custom")
            >>> spa.get_emitter("custom")    # access using given alias

            Add emitter relative to microphone
            >>> spa.add_emitter(position=[0.1, 0.1, 0.1], alias="custom", mic="tester")
            >>> spa.get_emitter("custom")

            Add emitter with a random position that is in a direct line with the microphone we placed above
            >>> spa.add_emitter(ensure_direct_path="tester")
        """
        # Remove existing emitters if we wish to do this
        if not keep_existing:
            self.clear_emitters()

        # Sanity checking
        if polar:
            assert mic is not None, "mic_alias is required for polar coordinates"

        # Parse the list of microphone aliases that we require a direct line to
        direct_path_to = self._parse_valid_microphone_aliases(ensure_direct_path)

        # If we want to express our emitters relative to a given microphone, grab this now
        desired_mic = self._get_mic_from_alias(mic)

        # Get the alias for this emitter
        alias = (
            utils.get_default_alias("src", self.emitters) if alias is None else alias
        )

        # Try and place inside the mesh: return True if placed, False if not
        placed = self._try_add_emitter(
            position, desired_mic, alias, polar, direct_path_to
        )

        # If we can't add the emitter to the mesh
        if not placed:
            # If we were trying to add it to a random position
            if position is None:
                raise ValueError(
                    f"Could not place emitter in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                    f"If this is happening frequently, consider reducing the number of `emitters`, "
                    f"or the `empty_space_around` arguments."
                )
            # If we were trying to add it to a specific position
            else:
                raise ValueError(
                    f"Position {position} invalid when placing emitter inside the mesh! "
                    f"If this is happening frequently, consider reducing the number of `emitters`, "
                    f"or the `empty_space_around` arguments."
                )

    @utils.update_state
    def add_emitters(
        self,
        positions: Union[list, np.ndarray, None] = None,
        aliases: list[str] = None,
        mics: Union[list[str], str] = None,
        n_emitters: Optional[int] = None,
        keep_existing: bool = False,
        polar: bool = True,
        ensure_direct_path: Union[bool, list, str, None] = False,
        raise_on_error: bool = True,
    ) -> None:
        """
        Add emitters to the mesh.

        This function essentially takes in lists of the arguments expected by `add_emitters`. The `raise_on_error`
        command will skip over microphones that cannot be placed in the mesh and raise a warning in the console.

        Additionally, `n_emitters` can be provided instead of `positions` to choose a number of emitters to add randomly.

        Arguments:
            positions: Locations to add the emitters, defaults to a single random location.
            aliases: String references to assign the emitters inside the `emitters` dictionary.
            mics: String references to microphones inside the `microphones` dictionary.
            keep_existing (optional): whether to keep existing emitters from the mesh or remove, defaults to keep.
            raise_on_error (optional): if True, raises an error when unable to place emitter, otherwise skips to next.
            n_emitters: Number of emitters to add with random positions
            polar (optional): if True, `position` is expected in form [azimuth, colatitude, elevation] relative to mic
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
        """
        # Remove existing emitters if we wish to do this
        if not keep_existing:
            self.clear_emitters()

        if polar:
            assert mics is not None, "mic_alias is required for polar coordinates"

        # Parse the list of microphone aliases that we require a direct line to
        direct_path_to = self._parse_valid_microphone_aliases(ensure_direct_path)

        if positions is not None and n_emitters is not None:
            raise TypeError("Cannot specify both `n_emitters` and `positions`.")

        if n_emitters is not None:
            assert isinstance(n_emitters, int), "`n_emitters` must be an integer!"
            assert n_emitters > 0, "`n_emitters` must be positive!"
            positions = [None for _ in range(n_emitters)]

        all_not_none = [
            l_
            for l_ in [positions, aliases, mics]
            if l_ is not None and isinstance(l_, (list, np.ndarray))
        ]
        # Handle cases where we haven't provided an equal number of positions and aliases
        if not utils.check_all_lens_equal(*all_not_none):
            raise ValueError("Expected all inputs to have equal length")

        # Get the index to iterate up to
        max_idx = max([len(a) for a in all_not_none]) if len(all_not_none) > 0 else 0
        # Tile the mic aliases if we've only provided a single one
        if isinstance(mics, str):
            mics = [mics for _ in range(max_idx)]

        # Iterate over all the emitters we want to place
        for idx in range(max_idx):
            position_ = positions[idx] if positions is not None else None
            emitter_alias_ = aliases[idx] if aliases is not None else None
            mic_alias_ = mics[idx] if mics is not None else None

            # If we want to express our emitters relative to a given microphone, grab this now
            desired_mic = self._get_mic_from_alias(mic_alias_)

            # Get the emitter alias
            emitter_alias_ = (
                utils.get_default_alias("src", self.emitters)
                if emitter_alias_ is None
                else emitter_alias_
            )

            # Try and place the emitter inside the space
            placed = self._try_add_emitter(
                position_, desired_mic, emitter_alias_, polar, direct_path_to
            )

            # If we can't add the emitter to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if position_ is None:
                    msg = (
                        f"Could not place emitter in the mesh after {utils.MAX_PLACE_ATTEMPTS} attempts. "
                        f"Consider reducing `empty_space_around` arguments."
                    )
                # If we were trying to add it to a specific position
                else:
                    msg = (
                        f"Position {position_} invalid for emitter. "
                        f"Consider reducing `empty_space_around` arguments."
                    )

                # Raise the error if required or just log a warning and skip to the next emitter
                if raise_on_error:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

    def define_trajectory(
        self,
        duration: Optional[utils.Numeric],
        starting_position: Optional[Union[np.ndarray, list]] = None,
        ending_position: Optional[Union[np.ndarray, list]] = None,
        max_speed: Optional[utils.Numeric] = MOVING_EMITTER_MAX_SPEED,
        temporal_resolution: Optional[
            utils.Numeric
        ] = MOVING_EMITTER_TEMPORAL_RESOLUTION,
        shape: Optional[str] = "linear",
        max_place_attempts: Optional[utils.Numeric] = utils.MAX_PLACE_ATTEMPTS,
    ):
        """
        Defines a trajectory for a moving sound event with specified spatial bounds and event duration.

        This method calculates a series of XYZ coordinates that outline the path of a sound event, based on the
        specified trajectory shape, the confines of the mesh, and the duration of the event. It generates a starting
        point and an end point that comply with these conditions, and then interpolates between these points according
        to the trajectory's shape.

        Arguments:
            duration (Numeric): the length of time it should take to traverse from starting to ending position
            starting_position (np.ndarray): the starting position for the trajectory. If not provided, a random valid
                position within the mesh will be selected.
            ending_position (np.ndarray): the ending position for the trajectory. If not provided, a random valid
                position within the mesh that has line-of-sight with `starting_position` will be selected.
            max_speed (Numeric): the speed limit for the trajectory, in meters per second
            temporal_resolution (Numeric): the number of emitters created per second
            shape (str): the shape of the trajectory; currently, only "linear" and "circular" are supported.
            max_place_attempts (Numeric): the number of times to try and create the trajectory.

        Raises:
            ValueError: if a trajectory cannot be defined after `max_place_attempts`

        Returns:
            np.ndarray: the sanitised trajectory, with shape (n_points, 3)
        """

        # If we've provided BOTH a starting and ending position, we should only try and define the trajectory once
        #  This is because the trajectory will always be deterministic (same on every attempt) with a known start/end,
        #  and we don't want to continue iterating unnecessarily when the outcome is already known
        actual_place_attempts = (
            max_place_attempts
            if starting_position is None or ending_position is None
            else 1
        )
        actual_place_attempts = int(
            utils.sanitise_positive_number(actual_place_attempts)
        )

        # Sanitise provided shape
        #  Only accept a linear shape for now
        accepteds = [
            "linear",
        ]
        if shape not in accepteds:
            raise ValueError(
                f"`shape` must be one of {', '.join(accepteds)} but got '{shape}'"
            )

        # Compute the number of samples based on duration and resolution
        n_points = int(
            utils.sanitise_positive_number(duration * temporal_resolution) + 1
        )
        max_distance = utils.sanitise_positive_number(max_speed * duration)

        # Compute the distance that we can travel in a single step
        step_limit = max_speed / temporal_resolution

        if max_distance < 1.0:
            logger.warning(
                f"Maximum trajectory distance is small ({max_distance:.2f} m). "
                f"If a valid trajectory cannot be created, consider increasing the duration, max_speed, "
                f"or relaxing spatial constraints."
            )

        # Try and create the trajectory a specified number of times
        for attempt in range(actual_place_attempts):

            # Log progress every 100 attempts
            if (attempt + 1) % 100 == 0:
                logger.info(f"Trajectory attempt {attempt + 1}/{actual_place_attempts}")

            # Use random starting and ending positions if not defined already: ending position assumes LOS with start
            start_attempt = (
                self.get_random_position()
                if starting_position is None
                else starting_position
            )
            end_attempt = (
                self.get_random_position()
                if ending_position is None
                else ending_position
            )

            # Sanitise starting and ending position
            start_attempt = utils.sanitise_coordinates(start_attempt)
            end_attempt = utils.sanitise_coordinates(end_attempt)

            # Continue if either starting or ending position is invalid
            if not all(
                (
                    self._validate_position(start_attempt),
                    self._validate_position(end_attempt),
                )
            ):
                continue

            # Reject if ending point is too far away from starting point
            distance = np.linalg.norm(end_attempt - start_attempt)
            if distance > max_distance:
                continue

            # Continue if no LOS exists between starting and ending position for a linear trajectory
            if shape == "linear" and not self.path_exists_between_points(
                start_attempt, end_attempt
            ):
                continue

            # Compute the trajectory with the utility function
            if shape == "linear":
                trajectory = utils.generate_linear_trajectory(
                    start_attempt, end_attempt, n_points
                )
            else:
                trajectory = utils.generate_circular_trajectory(
                    start_attempt, end_attempt, n_points
                )

            # Ensure no step exceeds max_speed / temporal_resolution
            deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            if np.any(deltas > step_limit + 1e-4):
                continue

            # Ensure we have at least two steps in the trajectory
            if len(trajectory) < 2:
                continue

            # Validate that all the positions in the trajectory are acceptable
            #  (in bounds of mesh, not too close to a microphone or another placed emitter, etc.)
            if self._validate_position(trajectory):
                return trajectory

        # If we reach here, we couldn't create the trajectory
        raise ValueError(
            f"Could not define a valid movement trajectory after {actual_place_attempts} attempt(s). Consider:\n"
            f"- Reducing `empty_space_around parameters`\n"
            f"- Decreasing `temporal_resolution` (currently {temporal_resolution})\n"
            f"- Increasing `max_place_attempts` (currently {max_place_attempts})\n"
        )

    def _simulation_sanity_check(self) -> None:
        """
        Check conditions required for simulation are met
        """
        assert (
            len(self.emitters) > 0
        ), "Must have added valid emitters to the mesh before calling `.simulate`!"
        assert (
            len(self.microphones) > 0
        ), "Must have added valid microphones to the mesh before calling `.simulate`!"
        assert all(
            type(m) in MICARRAY_LIST for m in self.microphones.values()
        ), "Non-microphone objects in microphone attribute"
        assert (
            self.ctx.get_listener_count() > 0
        ), "Must have listeners added to the ray tracing engine"
        assert (
            self.ctx.get_source_count() > 0
        ), "Must have emitters added to the ray tracing engine"
        # Check we have the expected number of sources and listeners
        assert len(self.emitters) == self.ctx.get_source_count()
        assert (
            sum(m.n_capsules for m in self.microphones.values())
            == self.ctx.get_listener_count()
        )

    def simulate(self) -> None:
        """
        Simulates audio propagation in the state with the current listener and sound emitter positions.
        """
        # Sanity check that we actually have emitters and microphones in the state
        self._simulation_sanity_check()
        # Clear out any existing IRs
        self._irs = None
        # Run the simulation
        self.ctx.simulate()
        efficiency = self.ctx.get_indirect_ray_efficiency()
        # Log the ray efficiency: outdoor would have a very low value, e.g. < 0.05.
        #  A closed indoor room would have >0.95, and a room with some holes might be in the 0.1-0.8 range.
        #  If the ray efficiency is low for an indoor environment, it indicates a lot of ray leak from holes.
        logger.info(
            f"Finished simulation! Overall indirect ray efficiency: {efficiency:.3f}"
        )
        if efficiency < WARN_WHEN_EFFICIENCY_BELOW:
            logger.warning(
                f"Ray efficiency is below {WARN_WHEN_EFFICIENCY_BELOW:.0%}. It is possible that the mesh "
                f"may have holes in it. Consider decreasing `repair_threshold` when initialising the "
                f"`WorldState` object, or running `trimesh.repair.fill_holes` on your mesh."
            )
        # Compute the IRs: this gives us shape (N_capsules, N_emitters, N_channels == 1, N_samples)
        irs = self.ctx.get_audio()
        # Format irs into a dictionary of {mic000: (N_capsules, N_emitters, N_samples), mic001: (...)}
        #  with one key-value pair per microphone. We have to do this because we cannot have ragged arrays
        #  The individual arrays can then be accessed by calling `self.irs.values()`
        self._irs = self._format_irs(irs)

    def _format_irs(self, irs: np.ndarray) -> dict:
        """
        Formats IRs from the ray tracing engine into a dictionary of {mic1: (N_capsules, N_emitters, N_samples), ...}
        """
        # Define a counter that we can use to access the flat array of (capsules, emitters, samples)
        counter = 0
        all_irs = OrderedDict()
        for mic_alias, mic in self.microphones.items():
            mic_ir = []
            # Iterate over the capsules associated with this microphone
            for n_capsule in range(counter, mic.n_capsules + counter):
                counter += 1
                # This just gets the mono audio for each capsule
                capsule_ir = irs[n_capsule, :, 0, :]
                mic_ir.append(capsule_ir)
            # Stack to a shape of (N_capsules, N_emitters, N_samples)
            mic.irs = np.stack(mic_ir)
            # Get the name of the mic and create a new key-value pair
            all_irs[mic_alias] = mic.irs
        return all_irs

    def create_scene(
        self, mic_radius: float = 0.2, emitter_radius: float = 0.1
    ) -> trimesh.Scene:
        """
        Creates a trimesh.Scene with the Space's mesh, microphone position, and emitters all added

        Returns:
            trimesh.Scene: The rendered scene, that can be shown in e.g. a notebook with the `.show()` command
        """
        scene = self.mesh.scene()
        # This just adds the microphone positions
        for mic in self.microphones.values():
            for capsule in mic.coordinates_absolute:
                add_sphere(scene, capsule, color=[255, 0, 0], r=mic_radius)
        # This adds the sound emitters, with different color + radius
        for emitter_list in self.emitters.values():
            for emitter in emitter_list:
                add_sphere(
                    scene, emitter.coordinates_absolute, [0, 255, 0], r=emitter_radius
                )
        return scene  # can then run `.show()` on the returned object

    def create_plot(
        self,
    ) -> plt.Figure:
        """
        Creates a matplotlib.Figure object corresponding to top-down and side-views of the scene

        Returns:
            plt.Figure: The rendered figure that can be shown with e.g. plt.show()
        """
        # Create a figure with two subplots side by side
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        vertices = self.mesh.vertices
        # Create a top-down view first, then a side view
        mic_positions = np.vstack(
            [m.coordinates_absolute for m in self.microphones.values()]
        )
        emitter_positions = np.vstack(
            [x.coordinates_absolute for xs in self.emitters.values() for x in xs]
        )
        for ax_, idx, color, ylab, title in zip(
            ax.flatten(), [1, 2], ["red", "blue"], ["Y", "Z"], ["Top", "Side"]
        ):
            # Scatter the vertices first
            ax_.scatter(vertices[:, 0], vertices[:, idx], c="gray", alpha=0.1, s=1)
            # Then the microphone and emitter positions
            ax_.scatter(
                mic_positions[:, 0],
                mic_positions[:, idx],
                c="red",
                s=100,
                label="Microphone",
            )
            ax_.scatter(
                emitter_positions[:, 0],
                emitter_positions[:, idx],
                c="blue",
                s=25,
                alpha=0.5,
                label="Emitters",
            )
            # These are just plot aesthetics
            ax_.set_xlabel("X")
            ax_.set_ylabel(ylab)
            ax_.set_title(f'{title} view of {self.mesh.metadata["fpath"]}')
            ax_.legend()
            ax_.axis("equal")
            ax_.grid(True)
        # Return the matplotlib figure object
        fig.tight_layout()
        return fig  # can be used with plt.show, fig.savefig, etc.

    def save_irs_to_wav(self, outdir: str) -> None:
        """
        Writes IRs to WAV audio files.

        IRs will be dumped in the form `mic{i1}_capsule{i2}_emitter_{i3}.wav`. For instance, with two emitters and two
        mono microphones, we'd expect `mic000_capsule000_emitter000.wav`, `mic001_capsule_000_emitter_001.wav`,
        `mic001_capsule_000_emitter_000.wav`, and `mic002_capsule000_emitter_002.wav`.

        Args:
            outdir (str): IRs will be saved here.
        """
        assert self._irs is not None, "IRs have not been created yet!"
        assert os.path.isdir(outdir), f"Output directory {outdir} does not exist!"
        # This iterates over [emitters, channels, samples]
        for mic_alias, mic in self.microphones.items():
            for caps_idx, caps in enumerate(mic.irs):
                caps_idx = str(caps_idx).zfill(3)
                for emitter_idx, emitter in enumerate(caps):
                    emitter_idx = str(emitter_idx).zfill(3)
                    fname = os.path.join(
                        outdir,
                        f"{mic_alias}_capsule{caps_idx}_emitter{emitter_idx}.wav",
                    )
                    # Dump the audio to a 16-bit PCM wav using our predefined sample rate
                    utils.write_wav(emitter, fname, self.ctx.config.sample_rate)

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        return dict(
            emitters={
                s_alias: [s_.to_dict() for s_ in s]
                for s_alias, s in self.emitters.items()
            },
            microphones={
                m_alias: m.to_dict() for m_alias, m in self.microphones.items()
            },
            mesh=dict(
                **self.mesh.metadata,  # this gets us the filepath, filename, and file extension of the mesh
                bounds=self.mesh.bounds.tolist(),
                centroid=self.mesh.centroid.tolist(),
            ),
            # Get all the keywords from the ray-tracing configuration
            rlr_config={
                name: getattr(self.ctx.config, name)
                for name in dir(self.ctx.config)
                if not name.startswith("__")
                and not callable(getattr(self.ctx.config, name))
            },
            empty_space_around_mic=self.empty_space_around_mic,
            empty_space_around_emitter=self.empty_space_around_emitter,
            empty_space_around_surface=self.empty_space_around_surface,
            empty_space_around_capsule=self.empty_space_around_capsule,
            repair_threshold=self.repair_threshold,
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate a `WorldState` from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `WorldState`.

        Returns:
            WorldState instance.
        """

        # Validate the input
        for k in ["emitters", "microphones", "mesh", "rlr_config"]:
            if k not in input_dict:
                raise KeyError(f"Missing key: '{k}'")

        # Instantiate the state
        state = cls(
            mesh=input_dict["mesh"]["fpath"],
            empty_space_around_mic=input_dict["empty_space_around_mic"],
            empty_space_around_emitter=input_dict["empty_space_around_emitter"],
            empty_space_around_surface=input_dict["empty_space_around_surface"],
            empty_space_around_capsule=input_dict["empty_space_around_capsule"],
            repair_threshold=input_dict["repair_threshold"],
            rlr_kwargs=input_dict["rlr_config"],
        )

        # Instantiate the microphones and emitters from their dictionaries
        state.microphones = OrderedDict(
            {a: MicArray.from_dict(v) for a, v in input_dict["microphones"].items()}
        )
        state.emitters = OrderedDict(
            {
                a: [Emitter.from_dict(v_) for v_ in v]
                for a, v in input_dict["emitters"].items()
            }
        )

        # Update the state so we add everything in to the ray-tracing engine
        state._update()

        return state

    def __len__(self) -> int:
        """
        Returns the number of objects in the mesh (i.e., number of microphones + emitters)
        """
        return len(self.microphones) + len(self.emitters)

    def __str__(self) -> str:
        """
        Returns a string representation of the WorldState
        """
        return (
            f"'WorldState' with mesh '{self.mesh.metadata['fpath']}' and "
            f"{len(self)} objects ({len(self.microphones)} microphones, {len(self.emitters)} emitters)"
        )

    def __repr__(self) -> str:
        """
        Returns a JSON-formatted string representation of the WorldState
        """
        return utils.repr_as_json(self)

    def __getitem__(self, alias: str) -> list[Emitter]:
        """
        An alternative for `self.get_emitters(alias) or `self.emitters[alias]`
        """
        return self.get_emitters(alias)

    def __eq__(self, other: Any):
        """
        Compare two WorldState objects for equality.

        Returns:
            bool: True if two WorldState objects are equal, False otherwise
        """

        # Non-Event objects are always not equal
        if not isinstance(other, WorldState):
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

    def get_emitter(self, alias: str, emitter_idx: int = 0) -> Emitter:
        """
        Given a valid alias and index, get a single `Emitter` object, as in `self.emitters[alias][emitter_idx]`
        """
        emitter_list = self.get_emitters(alias)
        try:
            return emitter_list[emitter_idx]
        except IndexError:
            raise IndexError(
                f"Could not get idx {emitter_idx} for a list of Emitters with length {len(emitter_list)}"
            )

    def get_emitters(self, alias: str) -> list[Emitter]:
        """
        Given a valid alias, get a list of associated `Emitter` objects, as in `self.emitters[alias]`
        """
        if alias in self.emitters.keys():
            return self.emitters[alias]
        else:
            raise KeyError("Emitter alias '{}' not found.".format(alias))

    def get_microphone(self, alias: str) -> Type["MicArray"]:
        """
        Given a valid alias, get an associated `Microphone` object, as in `self.microphones[alias]`.
        """
        if alias in self.microphones.keys():
            return self.microphones[alias]
        else:
            raise KeyError("Microphone alias '{}' not found.".format(alias))

    @utils.update_state
    def clear_microphones(self) -> None:
        """
        Removes all current microphones.
        """
        self.microphones = OrderedDict()

    @utils.update_state
    def clear_emitters(self) -> None:
        """
        Removes all current emitters.
        """
        self.emitters = OrderedDict()

    @utils.update_state
    def clear_microphone(self, alias: str) -> None:
        """
        Given an alias for a microphone, clear that microphone if it exists and update the state.
        """
        if alias in self.microphones.keys():
            del self.microphones[alias]
        else:
            raise KeyError("Microphone alias '{}' not found.".format(alias))

    @utils.update_state
    def clear_emitter(self, alias: str) -> None:
        """
        Given an alias for an emitter, clear that emitter and update the state.
        """
        if alias in self.emitters.keys():
            del self.emitters[alias]
        else:
            raise KeyError("Emitter alias '{}' not found.".format(alias))
