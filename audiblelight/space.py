#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for representing triangular meshes, handling spatial operations, generating RIRs."""

import os
from pathlib import Path
from typing import Any, Union, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from rlr_audio_propagation import Config, Context, ChannelLayout, ChannelLayoutType
from loguru import logger

from audiblelight import utils
from audiblelight.micarrays import get_micarray_from_string, MICARRAY_LIST, MicArray, MonoCapsule

FACE_FILL_COLOR = [255, 0, 0, 255]

MIN_AVG_RAY_LENGTH = 3.0
MAX_PLACE_ATTEMPTS = 100    # Max number of times we'll attempt to place a source or microphone before giving up

MIN_DISTANCE_FROM_SOURCE = 0.2  # Minimum distance one sound source can be from another
MIN_DISTANCE_FROM_MIC = 0.1    # Minimum distance one sound source can be from the mic
MIN_DISTANCE_FROM_SURFACE = 0.2    # Minimum distance from the nearest mesh surface

WARN_WHEN_EFFICIENCY_BELOW = 0.5    # when the ray efficiency is below this value, raise a warning in .simulate


def load_mesh(mesh: Union[str, Path]) -> trimesh.Trimesh:
    """
    Loads a mesh from disk and coerces units to meters
    """
    # Passed in filepath as a string: convert to a Path
    if isinstance(mesh, (str, Path)):
        # Coerce string types to Path
        if isinstance(mesh, str):
            mesh = Path(mesh)
        # Raise a nicer error when the file can't be found
        if not mesh.is_file():
            raise FileNotFoundError(f"Cannot find mesh file at {mesh}, does it exist?")
        # Load up in trimesh, setting the metadata dictionary nicely
        #  This just allows us to access the filename, etc., later
        metadata = dict(fname=mesh.stem, ftype=mesh.suffix, fpath=str(mesh))
        # noinspection PyTypeChecker
        loaded_mesh = trimesh.load_mesh(mesh, file_type=mesh.suffix, metadata=metadata)
        # Convert the units of the mesh to meters, if this is not provided
        if loaded_mesh.units != utils.MESH_UNITS:
            logger.warning(f"Mesh {mesh.stem} has units {loaded_mesh.units}, converting to {utils.MESH_UNITS}")
            loaded_mesh = loaded_mesh.convert_units(utils.MESH_UNITS, guess=True)
    # Passed in something else
    else:
        raise TypeError(f"Expected mesh to be either a filepath or Trimesh object, but got {type(mesh)}")
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


def add_sphere(scene: trimesh.Scene, pos: np.array, color: list[int] = None, r: float = 0.2) -> None:
    """Adds a sphere object to a scene with given position, color, and radius"""
    if color is None:
        color = [0, 0, 0]
    sphere = trimesh.creation.uv_sphere(radius=r)
    sphere.apply_translation(pos)
    sphere.visual.face_colors = color
    scene.add_geometry(sphere)


class Space:
    """
    Represents a 3D space defined by a mesh, microphone position(s), and source position(s)

    This class is capable of handling spatial operations and simulating audio propagation using the ray-tracing library.

    Attributes:
        mesh (str, Path, trimesh.Trimesh): The mesh, either loaded as a trimesh or a path to a glb object on the disk.
        microphones (np.array): Position of the microphone in the mesh.
        ctx (rlr_audio_propagation.Context): The context for audio propagation simulation.
        sources (np.array): relative positions of sound sources

    """
    def __init__(
            self,
            mesh: Union[str, trimesh.Trimesh],
            min_distance_from_mic: float = MIN_DISTANCE_FROM_MIC,
            min_distance_from_source: float = MIN_DISTANCE_FROM_SOURCE,
            min_distance_from_surface: float = MIN_DISTANCE_FROM_SURFACE,
            repair_threshold: Optional[float] = None
    ):
        """
        Initializes the Space with a mesh and optionally a specific microphone position, and sets up the audio context.

        Args:
            mesh (str|trimesh.Trimesh): The name of the mesh file. Units will be coerced to meters when loading
            min_distance_from_mic (float): minimum meters new sources/mics will be placed from other mics
            min_distance_from_source (float): minimum meters new sources/mics will be placed from other sources
            min_distance_from_surface (float): minimum meters new sources/mics will be placed from mesh sources
            repair_threshold (float, optional): when the proportion of broken faces on the mesh is below this value,
                repair the mesh and fill holes. If None, will never repair the mesh.
        """
        # Store source and mic positions in here to access later; these should be in ABSOLUTE form
        self.sources = {}
        self.microphones = {}
        self._irs = None    # will be updated when calling `simulate`

        # Distances from objects/mesh surfaces
        self.min_distance_from_mic = min_distance_from_mic
        self.min_distance_from_surface = min_distance_from_surface
        self.min_distance_from_source = min_distance_from_source

        # Load in the trimesh object
        self.mesh = load_mesh(mesh)
        # If we want to try and repair the mesh, and if it actually needs repairing
        if repair_threshold is not None and not self.mesh.is_watertight:
            # Get the idxs of faces in the mesh that break the watertight status
            broken_faces = get_broken_faces(self.mesh)    # this uses copies so nothing will be set in-place
            # If the proportion of broken faces is below the desired threshold, do the repair in-place
            if len(broken_faces) / self.mesh.faces.shape[0] < repair_threshold:
                repair_mesh(self.mesh)

        # Setting up audio context
        # TODO: is it possible to set the sample rate here?
        cfg = Config()
        self.ctx = Context(cfg)
        self._setup_audio_context()

    @property
    def irs(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of IRs in the shape {mic000: (N_capsules, N_sources, N_samples), mic001: (...)}
        """
        if self._irs is None:
            raise AttributeError("IRs have not been simulated yet: add microphones and sources and call `simulate`.")
        else:
            return self._irs

    def calculate_weighted_average_ray_length(self, point: np.ndarray, num_rays: int = 100) -> float:
        """
        Calculate the weighted average length of rays cast from a point into a mesh.
        """
        # Generate random azimuthal angles for each ray
        angles = np.random.uniform(0, 2 * np.pi, num_rays)
        # Generate random elevation angles for each ray
        elevations = np.random.uniform(-np.pi / 2, np.pi / 2, num_rays)
        # Convert spherical coordinates (angles, elevations) to  Cartesian 3D direction vectors
        directions = np.column_stack([
            np.cos(elevations) * np.cos(angles),  # X component
            np.cos(elevations) * np.sin(angles),  # Y component
            np.sin(elevations)  # Z component
        ])
        # Repeat the origin point for each ray so rays start from the same position
        origins = np.tile(point, (num_rays, 1))
        # Cast rays from the origin in the computed directions and find the longest intersection distances with the mesh
        distances = trimesh.proximity.longest_ray(self.mesh, origins, directions)
        # We can get `inf` values here, likely due to holes in the mesh causing a ray never to intersect
        if any(np.isinf(distances)):
            # For simplicity, we can just remove these here but raise a warning
            logger.warning(f"Some rays cast from point {point} have infinite distances: is the mesh watertight?")
            distances = distances[distances != np.inf]
        # Compute weights by squaring the distances to give more importance to longer rays
        weights = distances ** 2
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

    def _setup_listener(self) -> None:
        """
        Adds a listener to the audio context and sets its position to the microphone's position.
        """
        # Stack the coordinates of all capsules into a single 2D array
        all_caps = np.vstack([m.coordinates_absolute for m in self.microphones.values()])
        # Iterate over all the capsules
        for caps_idx, caps_pos in enumerate(all_caps):  # type: np.ndarray
            # Add a single listener for each individual capsule
            self.ctx.add_listener(ChannelLayout(ChannelLayoutType.Mono, 1))
            self.ctx.set_listener_position(caps_idx, caps_pos.tolist())

    @staticmethod
    def _sanitize_microphone_input(microphone_type: Any) -> Type['MicArray']:
        """
        Sanitizes any microphone input into the correct 'MicArray' class.

        Returns:
            Type['MicArray']: the sanitized microphone class, ready to be instantiated
        """

        # Parsing the microphone type
        # If None, get a random microphone and use a randomized position
        if microphone_type is None:
            logger.warning(f"No microphone type provided, using a mono microphone capsule in a random position!")
            # Get a random microphone class
            sanitized_microphone = MonoCapsule

        # If a string, use the desired microphone type but get a random position
        elif isinstance(microphone_type, str):
            sanitized_microphone = get_micarray_from_string(microphone_type)

        # If a class contained inside MICARRAY_LIST
        elif microphone_type in MICARRAY_LIST:
            sanitized_microphone = microphone_type

        # Otherwise, we don't know what the microphone is
        else:
            raise TypeError(f"Could not parse microphone type {type(microphone_type)}")

        return sanitized_microphone

    def _try_add_microphone(self, mic_cls, position: Union[list, None], alias: str) -> bool:
        """
        Try to place a microphone of type mic_cls at position with given alias. Return True if successful.
        """
        if alias in self.microphones.keys():
            raise KeyError(f"Alias {alias} already exists in microphone dictionary")

        for attempt in range(MAX_PLACE_ATTEMPTS):
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

    def _get_default_microphone_alias(self) -> str:
        """Returns a default alias for a microphone"""
        n_current_mics = len(self.microphones) + 1 if len(self.microphones) > 0 else 0
        return f"mic{str(n_current_mics).zfill(3)}"

    def _clear_microphones(self) -> None:
        """Removes all current microphones"""
        self.microphones = {}
        self.ctx.clear_listeners()

    def add_microphone(
            self,
            microphone_type: Union[str, Type['MicArray'], None] = None,
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
            Create a space with a given mesh
            >>> spa = Space(mesh=...)

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

        # Remove existing microphones if we wish to do this
        if not keep_existing:
            self._clear_microphones()

        # Get the correct microphone type.
        sanitized_microphone = self._sanitize_microphone_input(microphone_type)

        # Get the microphone alias
        alias = self._get_default_microphone_alias() if alias is None else alias

        # Try and place the microphone inside the space
        placed = self._try_add_microphone(sanitized_microphone, position, alias)

        # If we can't add the microphone to the mesh
        if not placed:
            # If we were trying to add it to a random position
            if position is None:
                raise ValueError(f"Could not place microphone in the mesh after {MAX_PLACE_ATTEMPTS} attempts. "
                                 f"Consider reducing `min_distance_from` arguments.")
            # If we were trying to add it to a specific position
            else:
                raise ValueError(f"Position {position} invalid for microphone {sanitized_microphone.name}. "
                                f"Consider reducing `min_distance_from` arguments.")

        # Set up the listeners inside the ray-tracing engine: add one mono listener per microphone capsule
        self._setup_listener()


    def add_microphones(
            self,
            microphone_types: list[Union[str, Type['MicArray'], None]] = None,
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
            Create a space with a given mesh
            >>> spa = Space(mesh=...)

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
            self._clear_microphones()

        # Handle cases with non-unique aliases
        if aliases is not None:
            if len(set(aliases)) != len(aliases):
                raise ValueError("Only unique aliases can be passed")

        all_not_none = [l for l in [microphone_types, positions, aliases] if l is not None]
        # Handle cases where we haven't provided an equal number of mic types, positions, and aliases
        if not utils.check_all_lens_equal(*all_not_none):
            raise ValueError("Expected all inputs to have equal length")

        # Get the index to iterate up to
        max_idx = max([len(a) for a in all_not_none]) if len(all_not_none) > 0 else 0
        # Iterate over all the microphones we want to place
        for idx in range(max_idx):
            microphone_type_ = microphone_types[idx] if microphone_types is not None else None
            position_ = positions[idx] if positions is not None else None
            alias_ = aliases[idx] if aliases is not None else None

            # Get the correct microphone type.
            sanitized_microphone = self._sanitize_microphone_input(microphone_type_)

            # Get the microphone alias
            alias_ = self._get_default_microphone_alias() if alias_ is None else alias_

            # Try and place the microphone inside the space
            placed = self._try_add_microphone(sanitized_microphone, position_, alias_)

            # If we can't add the microphone to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if position_ is None:
                    msg = (f"Could not place microphone in the mesh after {MAX_PLACE_ATTEMPTS} attempts. "
                           f"Consider reducing `min_distance_from` arguments.")
                # If we were trying to add it to a specific position
                else:
                    msg = (f"Position {position_} invalid for microphone {sanitized_microphone.name}. "
                           f"Consider reducing `min_distance_from` arguments.")

                # Raise the error if required or just log a warning and skip to the next microphone
                if raise_on_error:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

        # Set up the listeners inside the ray-tracing engine: add one mono listener per microphone capsule
        self._setup_listener()

    def get_random_position(self) -> np.ndarray:
        """
        Get a random position to place a source inside the mesh
        """
        # Get an initial microphone position
        mic_pos = self.get_random_point_inside_mesh()
        # Start iterating until we get an acceptable position
        for attempt in range(MAX_PLACE_ATTEMPTS):
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
            logger.error(f"Could not find a suitable position after {MAX_PLACE_ATTEMPTS} attempts. "
                         f"Using the last attempted position, which is {mic_pos}.")
        return mic_pos

    def get_random_point_inside_mesh(self) -> np.ndarray:
        """
        Generates a random valid point inside the mesh.

        Returns:
            np.array: A valid point within the mesh in XYZ format
        """
        while True:
            point = np.random.uniform(self.mesh.bounds[0], self.mesh.bounds[1])
            # This checks e.g. distance from surface, other sources, other mics, and that point is in-bounds
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

    def _setup_sources(self):
        """
        Sets the positions of sound sources in the space.
        """
        # Now, iterate only through the valid sources and add these to the mesh
        for i, pos in enumerate(self.sources.values()):
            self.ctx.add_source()
            self.ctx.set_source_position(i, pos.tolist() if isinstance(pos, np.ndarray) else pos)

    def _validate_position(self, pos_abs: np.ndarray) -> bool:
        """
        Validates a position with respect to the mesh and objects (microphones, sources) inside it
        """
        return all((
            # source must be a minimum distance from all other sources
            all(np.linalg.norm(pos_abs - pos) >= self.min_distance_from_source for pos in self.sources.values()),
            # source must be a minimum distance from every microphone capsule
            #  setting axis=1 means that we calculate the distance independently for every capsule on every microphone
            all(all(np.linalg.norm(pos_abs - mic.coordinates_absolute, axis=1) >= self.min_distance_from_mic) for mic in self.microphones.values()),
            # source must be a minimum distance from the surface
            bool(self.mesh.nearest.on_surface([pos_abs])[1][0] >= self.min_distance_from_surface),
            # source must be inside the mesh
            self._is_point_inside_mesh(pos_abs)
        ))

    def _try_add_source(
            self,
            position: Union[list, None],
            relative_mic: Optional[Type['MicArray']],
            source_alias: str,
            polar: bool
    ) -> bool:
        """
        Try to place a source at position with the given alias. Return True if successful, False otherwise.
        """
        for attempt in range(MAX_PLACE_ATTEMPTS):
            # Grab a random position for the source if required
            pos = position if position is not None else self.get_random_position()
            assert len(pos) == 3, f"Expected three coordinates but got {len(pos)}"
            # Pos will be in cartesian form, but we can convert to polar if required
            if polar:
                assert relative_mic is not None, "Must provide a reference mic to use polar coordinates"
                assert position is not None, "Must set polar to False when using random positions"
                pos = utils.polar_to_cartesian(pos)[0]
            # If we want to express the position relative to a given microphone
            if relative_mic is not None:
                # Add the source position to the center of the microphone
                pos = relative_mic.coordinates_center + pos
            # If we have a valid position for the source
            if self._validate_position(pos):
                self.sources[source_alias] = np.asarray(pos)
                return True
            # If we were trying to place the source in a specific location, only make one attempt at placing it
            elif position is not None:
                break
        return False

    def _get_default_source_alias(self) -> str:
        """Returns a default alias for a source"""
        n_current_sources = len(self.sources) + 1 if len(self.sources) > 0 else 0
        return f"src{str(n_current_sources).zfill(3)}"

    def _get_mic_from_alias(self, mic_alias: Optional[str]) -> Optional[Type['MicArray']]:
        """Get a given `MicArray` object from its alias"""
        if mic_alias is not None:
            if mic_alias not in self.microphones:
                raise KeyError(f"No microphone found with alias {mic_alias}!")
            return self.microphones[mic_alias]
        else:
            return None

    def _clear_sources(self) -> None:
        """Removes all current sources"""
        self.sources = {}
        self.ctx.clear_sources()

    def add_source(
            self,
            position: Union[list, np.ndarray, None] = None,
            source_alias: str = None,
            mic_alias: str = None,
            keep_existing: bool = False,
            polar: bool = True,
    ) -> None:
        """
        Add a source to the space.

        If `mic_alias` is a key inside `microphones`, `position` is assumed to be relative to that microphone; else,
        it is assumed to be in absolute terms. If `polar` is True, `position` should be in the form
        (azimuth°, polar°, radius); else, it should be in cartesian coordinates in meters with the form [x, y, z].
        Note that `mic_alias` must not be None when `polar` is True.

        Arguments:
            position: Location to add the source, defaults to a random, valid location.
            source_alias: String reference to access the source inside the `self.sources` dictionary.
            mic_alias: String reference to a microphone inside `self.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            keep_existing (optional): Whether to keep existing sources from the mesh or remove, defaults to keep
            polar: When True, expects `position` to be provided in [azimuth, colatitude, elevation] form; otherwise,
                units are [x, y, z] in absolute, cartesian terms.

        Examples:
            Create a space with a given mesh and add a microphone
            >>> spa = Space(mesh=...)
            >>> spa.add_microphone(alias="tester")

            Add a single source with a random position
            >>> spa.add_source()
            >>> spa.sources["src000"]    # access with default alias

            Add source with given position and alias
            >>> spa.add_source(position=[0.5, 0.5, 0.5], source_alias="custom")
            >>> spa.sources["custom"]    # access using given alias

            Add source relative to microphone
            >>> spa.add_source(position=[0.1, 0.1, 0.1], source_alias="custom", mic_alias="tester")
            >>> spa.sources["custom"]
        """
        # TODO: allow sources to be specified in polar units
        # Remove existing sources if we wish to do this
        if not keep_existing:
            self._clear_sources()

        if polar:
            assert mic_alias is not None, "mic_alias is required for polar coordinates"

        # If we want to express our sources relative to a given microphone, grab this now
        desired_mic = self._get_mic_from_alias(mic_alias)

        # Get the alias for this source
        source_alias = self._get_default_source_alias() if source_alias is None else source_alias

        # Try and place inside the mesh: return True if placed, False if not
        placed = self._try_add_source(position, desired_mic, source_alias, polar)
        # If we can't add the source to the mesh
        if not placed:
            # If we were trying to add it to a random position
            if position is None:
                raise ValueError(f"Could not place source in the mesh after {MAX_PLACE_ATTEMPTS} attempts. "
                                 f"If this is happening frequently, consider reducing the number of `sources`, "
                                 f"`min_distance_from_mic`, or `min_distance_from_source`.")
            # If we were trying to add it to a specific position
            else:
                raise ValueError(f"Position {position} invalid when placing source inside the mesh! "
                                 f"If this is happening frequently, consider reducing the number of `sources`, "
                                 f"`min_distance_from_mic`, or `min_distance_from_source`.")

        # Add the sources to the ray-tracing engine
        self._setup_sources()

    def add_sources(
            self,
            positions: Union[list, np.ndarray, None] = None,
            source_aliases: list[str] = None,
            mic_aliases: Union[list[str], str] = None,
            n_sources: Optional[int] = None,
            keep_existing: bool = False,
            polar: bool = True,
            raise_on_error: bool = True,
    ) -> None:
        """
        Add sources to the mesh.

        This function essentially takes in lists of the arguments expected by `add_sources`. The `raise_on_error`
        command will skip over microphones that cannot be placed in the mesh and raise a warning in the console.

        Additionally, `n_sources` can be provided instead of `positions` to choose a number of sources to add randomly.

        Arguments:
            positions: Locations to add the sources, defaults to a single random location.
            source_aliases: String references to assign the sources inside the `sources` dictionary.
            mic_aliases: String references to microphones inside the `microphones` dictionary.
            keep_existing (optional): whether to keep existing sources from the mesh or remove, defaults to keep.
            raise_on_error (optional): if True, raises an error when unable to place source, otherwise skips to next.
            n_sources: Number of sources to add with random positions
            polar (optional): if True, `position` is expected in form [azimuth, colatitude, elevation] relative to mic
        """
        # TODO: allow sources to be specified in polar units
        # Remove existing sources if we wish to do this
        if not keep_existing:
            self.sources = {}
            self.ctx.clear_sources()

        if polar:
            assert mic_aliases is not None, "mic_alias is required for polar coordinates"

        if positions is not None and n_sources is not None:
            raise TypeError("Cannot specify both `n_sources` and `positions`.")

        if n_sources is not None:
            assert isinstance(n_sources, int), "`n_sources` must be an integer!"
            assert n_sources > 0, "`n_sources` must be positive!"
            positions = [None for _ in range(n_sources)]

        # Handle cases with non-unique aliases
        if source_aliases is not None:
            if len(set(source_aliases)) != len(source_aliases):
                raise ValueError("Only unique aliases can be passed")

        all_not_none = [l for l in [positions, source_aliases, mic_aliases]
                        if l is not None and isinstance(l, (list, np.ndarray))]
        # Handle cases where we haven't provided an equal number of positions and aliases
        if not utils.check_all_lens_equal(*all_not_none):
            raise ValueError("Expected all inputs to have equal length")

        # Get the index to iterate up to
        max_idx = max([len(a) for a in all_not_none]) if len(all_not_none) > 0 else 0
        # Tile the mic aliases if we've only provided a single one
        if isinstance(mic_aliases, str):
            mic_aliases = [mic_aliases for _ in range(max_idx)]

        # Iterate over all the sources we want to place
        for idx in range(max_idx):
            position_ = positions[idx] if positions is not None else None
            source_alias_ = source_aliases[idx] if source_aliases is not None else None
            mic_alias_ = mic_aliases[idx] if mic_aliases is not None else None

            # If we want to express our sources relative to a given microphone, grab this now
            desired_mic = self._get_mic_from_alias(mic_alias_)

            # Get the source alias
            source_alias_ = self._get_default_source_alias() if source_alias_ is None else source_alias_

            # Try and place the source inside the space
            placed = self._try_add_source(position_, desired_mic, source_alias_, polar)

            # If we can't add the source to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if position_ is None:
                    msg = (f"Could not place source in the mesh after {MAX_PLACE_ATTEMPTS} attempts. "
                           f"Consider reducing `min_distance_from` arguments.")
                # If we were trying to add it to a specific position
                else:
                    msg = (f"Position {position_} invalid for source. "
                           f"Consider reducing `min_distance_from` arguments.")

                # Raise the error if required or just log a warning and skip to the next source
                if raise_on_error:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

        # Set up the sources in the ray-tracing engine
        self._setup_sources()

    def _simulation_sanity_check(self) -> None:
        """
        Check conditions required for simulation are met
        """
        assert len(self.sources) > 0, "Must have added valid sources to the mesh before calling `.simulate`!"
        assert len(self.microphones) > 0, "Must have added valid microphones to the mesh before calling `.simulate`!"
        assert all(type(m) in MICARRAY_LIST for m in self.microphones.values()), "Non-microphone objects in microphone attribute"
        assert self.ctx.get_listener_count() > 0, "Must have listeners added to the ray tracing engine"
        assert self.ctx.get_source_count() > 0, "Must have sources added to the ray tracing engine"

    def simulate(self) -> None:
        """
        Simulates audio propagation in the space with the current listener and source positions.
        """
        # Sanity check that we actually have sources and microphones in the space
        self._simulation_sanity_check()
        # Clear out any existing IRs
        self._irs = None
        # Run the simulation
        self.ctx.simulate()
        efficiency = self.ctx.get_indirect_ray_efficiency()
        # Log the ray efficiency: outdoor would have a very low value, e.g. < 0.05.
        #  A closed indoor room would have >0.95, and a room with some holes might be in the 0.1-0.8 range.
        #  If the ray efficiency is low for an indoor environment, it indicates a lot of ray leak from holes.
        logger.info(f"Finished simulation! Overall indirect ray efficiency: {efficiency:.3f}")
        if efficiency < WARN_WHEN_EFFICIENCY_BELOW:
            logger.warning(f"Ray efficiency is below {WARN_WHEN_EFFICIENCY_BELOW:.0%}. It is possible that the mesh "
                           f"may have holes in it. Consider decreasing `repair_threshold` when initialising the "
                           f"`Space` object, or running `trimesh.repair.fill_holes` on your mesh.")
        # Compute the IRs: this gives us shape (N_capsules, N_sources, N_channels == 1, N_samples)
        irs = self.ctx.get_audio()
        # Format irs into a dictionary of {mic000: (N_capsules, N_sources, N_samples), mic001: (...)}
        #  with one key-value pair per microphone. We have to do this because we cannot have ragged arrays
        #  The individual arrays can then be accessed by calling `self.irs.values()`
        self._irs = self._format_irs(irs)

    def _format_irs(self, irs: np.ndarray) -> dict:
        """
        Formats IRs from the ray tracing engine into a dictionary of {mic1: (N_capsules, N_sources, N_samples), ...}
        """
        # Define a counter that we can use to access the flat array of (capsules, sources, samples)
        counter = 0
        all_irs = {}
        for mic_alias, mic in self.microphones.items():
            mic_ir = []
            # Iterate over the capsules associated with this microphone
            for n_capsule in range(counter, mic.n_capsules + counter):
                counter += 1
                # This just gets the mono audio for each capsule
                capsule_ir = irs[n_capsule, :, 0, :]
                mic_ir.append(capsule_ir)
            # Stack to a shape of (N_capsules, N_sources, N_samples)
            mic.irs = np.stack(mic_ir)
            # Get the name of the mic and create a new key-value pair
            all_irs[mic_alias] = mic.irs
        return all_irs

    def create_scene(self, mic_radius: float = 0.2, source_radius: float = 0.1) -> trimesh.Scene:
        """
        Creates a trimesh.Scene with the Space's mesh, microphone position, and sources all added

        Returns:
            trimesh.Scene: The rendered scene, that can be shown in e.g. a notebook with the `.show()` command
        """
        scene = self.mesh.scene()
        # This just adds the microphone positions
        for mic in self.microphones.values():
            for capsule in mic.coordinates_absolute:
                add_sphere(scene, capsule, color=[255, 0, 0], r=mic_radius)
        # This adds the sound sources, with different color + radius
        for source in self.sources.values():
            add_sphere(scene, source, [0, 255, 0], r=source_radius)
        return scene    # can then run `.show()` on the returned object

    def create_plot(self, ) -> plt.Figure:
        """
        Creates a matplotlib.Figure object corresponding to top-down and side-views of the scene

        Returns:
            plt.Figure: The rendered figure that can be shown with e.g. plt.show()
        """
        # Create a figure with two subplots side by side
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        vertices = self.mesh.vertices
        # Create a top-down view first, then a side view
        mic_positions = np.vstack([m.coordinates_absolute for m in self.microphones.values()])
        source_positions = np.vstack([v for v in self.sources.values()])
        for ax_, idx, color, ylab, title in zip(ax.flatten(), [1, 2], ["red", "blue"], ["Y", "Z"], ["Top", "Side"]):
            # Scatter the vertices first
            ax_.scatter(vertices[:, 0], vertices[:, idx], c='gray', alpha=0.1, s=1)
            # Then the microphone and source positions
            ax_.scatter(mic_positions[:, 0], mic_positions[:, idx], c='red', s=100, label='Microphone')
            ax_.scatter(source_positions[:, 0], source_positions[:, idx], c='blue', s=25, alpha=0.5, label='Sources')
            # These are just plot aesthetics
            ax_.set_xlabel('X')
            ax_.set_ylabel(ylab)
            ax_.set_title(f'{title} view of {self.mesh.metadata["fpath"]}')
            ax_.legend()
            ax_.axis('equal')
            ax_.grid(True)
        # Return the matplotlib figure object
        fig.tight_layout()
        return fig    # can be used with plt.show, fig.savefig, etc.

    def save_irs_to_wav(self, outdir: str) -> None:
        """
        Writes IRs to WAV audio files.

        IRs will be dumped in the form `mic{i1}_capsule{i2}_source_{i3}.wav`. For instance, with two sources and two
        mono microphones, we'd expect `mic000_capsule000_source000.wav`, `mic001_capsule_000_source_001.wav`,
        `mic001_capsule_000_source_000.wav`, and `mic002_capsule000_source_002.wav`.

        Args:
            outdir (str): IRs will be saved here.
        """
        assert self._irs is not None, f"IRs have not been created yet!"
        assert os.path.isdir(outdir), f"Output directory {outdir} does not exist!"
        # This iterates over [sources, channels, samples]
        for mic_alias, mic in self.microphones.items():
            for caps_idx, caps in enumerate(mic.irs):
                caps_idx = str(caps_idx).zfill(3)
                for source_idx, source in enumerate(caps):
                    source_idx = str(source_idx).zfill(3)
                    fname = os.path.join(outdir, f"{mic_alias}_capsule{caps_idx}_source{source_idx}.wav")
                    # Dump the audio to a 16-bit PCM wav using our predefined sample rate
                    utils.write_wav(source, fname, self.ctx.config.sample_rate)
