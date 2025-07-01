#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for representing triangular meshes, handling spatial operations, generating RIRs."""

import os
import time
import random
from pathlib import Path
from typing import Union, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from rlr_audio_propagation import Config, Context, ChannelLayout, ChannelLayoutType
from netCDF4 import Dataset
from loguru import logger
from scipy.io import wavfile

from audiblelight import utils
from audiblelight.micarrays import get_micarray_from_string, MICARRAY_LIST, MicArray

FACE_FILL_COLOR = [255, 0, 0, 255]

MIN_AVG_RAY_LENGTH = 3.0
MAX_PLACE_ATTEMPTS = 100    # Max number of times we'll attempt to place a source or microphone before giving up

MIN_DISTANCE_FROM_SOURCE = 0.2  # Minimum distance one sound source can be from another
MIN_DISTANCE_FROM_MIC = 0.1    # Minimum distance one sound source can be from the mic
MIN_DISTANCE_FROM_SURFACE = 0.2    # Minimum distance from the nearest mesh surface

WARN_WHEN_EFFICIENCY_BELOW = 0.5    # when the ray efficiency is below this value, raise a warning in .simulate


def load_mesh(mesh: Union[str, Path, trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    Loads a mesh from disk or directly from a `trimesh.Trimesh` object and coerces units to meters
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
        loaded_mesh = trimesh.load_mesh(mesh, file_type=mesh.suffix, metadata=metadata)
        # Convert the units of the mesh to meters, if this is not provided
        if loaded_mesh.units != utils.MESH_UNITS:
            logger.warning(f"Mesh {mesh.stem} has units {loaded_mesh.units}, converting to {utils.MESH_UNITS}")
            loaded_mesh = loaded_mesh.convert_units(utils.MESH_UNITS, guess=True)
    # Passed in a loaded mesh object
    elif isinstance(mesh, trimesh.Trimesh):
        loaded_mesh = mesh
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
        mesh (str | Path | trimesh.Trimesh): The mesh, either loaded as a trimesh or a path to a glb object on the disk.
        microphones (np.array): Position of the microphone in the mesh.
        ctx (rlr_audio_propagation.Context): The context for audio propagation simulation.
        source_positions (np.array): relative positions of sound sources

    """
    def __init__(
            self,
            mesh: str | trimesh.Trimesh,
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
        self.source_positions = []
        self.microphones = []
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
        all_caps = np.vstack([m.coordinates_absolute for m in self.microphones])
        # Iterate over all the capsules
        for caps_idx, caps_pos in enumerate(all_caps):  # type: np.ndarray
            # Add a single listener for each individual capsule
            self.ctx.add_listener(ChannelLayout(ChannelLayoutType.Mono, 1))
            self.ctx.set_listener_position(caps_idx, caps_pos.tolist())

    @staticmethod
    def _sanitize_microphone_input(microphones) -> list[tuple]:
        """
        Sanitizes any microphone input into the form [(mic_type, mic_location), (mic_type, mic_location), (...)].

        Microphone types are coerced to their corresponding classes from `micarrays` and mic_locations are either
        1D arrays of coordinates in the form XYZ or None (in which case a random position will be assigned).

        Returns:
            list[tuple]: the sanitized microphone inputs, in form [(mic1_cls, mic1_location), (mic2_cls, mic2_location)]
        """
        # Convert single strings to list of strings
        sanitized_microphones = []

        # If None, get a random microphone and use a randomized position
        if microphones is None:
            logger.warning(f"No microphone positions provided, using a random microphone array in a random position!")
            # Get a random microphone class
            mic_cls = random.choice(MICARRAY_LIST)
            sanitized_microphones.append((mic_cls, None))

        # If a string, use the desired microphone type but get a random position
        elif isinstance(microphones, str):
            sanitized_microphones.append((get_micarray_from_string(microphones), None))

        # If a class contained inside MICARRAY_LIST
        elif microphones in MICARRAY_LIST:
            sanitized_microphones.append((microphones, None))

        # If an integer, assume that this is the number of random microphones we want to place
        elif isinstance(microphones, int):
            assert microphones > 0, f"Number of microphones to create must be greater than 0, but got {microphones}"
            for mic_idx in range(microphones):
                # Get a random microphone class
                mic_cls = random.choice(MICARRAY_LIST)
                sanitized_microphones.append((mic_cls, None))

        # If a dictionary of microphone types and positions
        elif isinstance(microphones, dict):
            assert len(microphones.items()) > 0, "Number of microphones to add must be greater than 0"
            # Raise a warning for this type of input
            logger.warning("Passing a dictionary of microphone types and coordinates is not recommended, as duplicates"
                           " will be removed. Instead, we recommend passing a list of tuples, where the first element"
                           " of each tuple is the desired microphone type and the second is the location to place it.")
            for mic_str, mic_pos in microphones.items():
                # Retrieve the class of microphone from the string name
                mic_cls = get_micarray_from_string(mic_str)
                sanitized_microphones.append((mic_cls, mic_pos))

        # If we've passed in an iterable
        elif isinstance(microphones, (list, np.ndarray)):
            assert len(microphones) > 0, "Number of microphones to add must be greater than 0"
            # Handle cases where we've just passed a single list of XYZ coordinates
            if len(microphones) == 3 and all(isinstance(i, (float, int)) for i in microphones):
                microphones = utils.coerce2d(microphones)

            # If the iterable is a list of strings, corresponding to microphone array names
            if all(isinstance(s, str) for s in microphones):
                for mic_str in microphones:
                    # We assume that this is the name of a microphone array
                    mic_cls = get_micarray_from_string(mic_str)
                    sanitized_microphones.append((mic_cls, None))

            # If the iterable contains tuples in the form (microphone_type, microphone_coords)
            elif all(
                    isinstance(item, (tuple, list)) and
                    len(item) == 2 and
                    isinstance(item[0], str) and
                    isinstance(item[1], (list, np.ndarray))
                    for item in microphones
            ):
                for mic_str, mic_pos in microphones:
                    # Retrieve the class of microphone from the string name
                    mic_cls = get_micarray_from_string(mic_str)
                    sanitized_microphones.append((mic_cls, mic_pos))

            # If the iterable contains lists of coordinates
            elif all(
                    isinstance(item, (tuple, list, np.ndarray)) and  # every element of list must be one of these types
                    len(item) == 3 and  # every element must be in XYZ form
                    all(isinstance(i, (float, int)) for i in item)  # all items in every element must be float or ints
                    for item in microphones
            ):
                for single_pos in microphones:
                    # Get a random microphone class
                    mic_cls = random.choice(MICARRAY_LIST)
                    sanitized_microphones.append((mic_cls, single_pos))

            # Otherwise, we don't know what the input is
            else:
                raise TypeError("Could not handle microphone input")

        # Raise when invalid input types encountered
        else:
            raise TypeError(f"Could not parse input with type {type(microphones)}")
        return sanitized_microphones

    def _try_add_microphone(self, mic_cls, position: Union[list, None]) -> bool:
        """
        Try to place a microphone of type mic_cls at position. Return True if successful, False otherwise.
        """
        for attempt in range(MAX_PLACE_ATTEMPTS):
            # Grab a random position for the microphone if required
            pos = position if position is not None else self.get_random_position()
            # Instantiate the microphone and set its coordinates
            mic = mic_cls()
            mic.set_absolute_coordinates(pos)
            # If we have a valid position for the microphone
            if all(self._validate_source_position(caps) for caps in mic.coordinates_absolute):
                self.microphones.append(mic)
                return True
            # If we were trying to place the microphone in a specific location, only make one attempt at placing it
            elif position is not None:
                break
        return False

    def add_microphones(
            self,
            microphones: Union[list, np.ndarray, dict, None, int, str] = 1,
            keep_existing: bool = False,
    ) -> None:
        """
        Add microphones to the mesh at valid positions. Must be called before `simulate`.

        Arguments:
            microphones (optional): the microphones to be added to the mesh.
                If an integer, will be interpreted as the number of microphones to add to the mesh, at random positions.
                If a string, must be the name of a valid microphone array.
                If a dictionary, should be in the format {mic_type: mic_center, mic_type: mic_center}, where mic_type
                 refers to one of the objects defined in `micarrays` and mic_center is a 1D array of cartesian
                 coordinates in form [X, Y, Z].
                 Note that dictionary types are not preferred over list of iterables, as dictionaries cannot have
                 duplicate keys, meaning that only one microphone of any type can be placed using this method.
                If a list, three input formats are accepted:
                 A list of strings, where each string is a microphone type to place in a random position
                 A list of iterables, where the first element is the microphone type and the second is its coordinates
                 A list of iterables, where each element is the coordinates for a random microphone
                If None, a single random microphone type will be added at a random position inside the mesh.
            keep_existing (optional): whether to remove existing microphones from the mesh, defaults to removing

        Examples:
            Create a space with a given mesh
            >>> spa = Space(mesh=...)

            Add a single microphone to the mesh with a random position
            >>> spa.add_microphones()

            Add three random microphones at three random positions
            >>> spa.add_microphones(3)

            Add a single Eigenmike32 at a random position
            >>> spa.add_microphones("eigenmike32")

            Add two AmbeoVR microphones at random positions
            >>> spa.add_microphones(["ambeovr", "ambeovr"])

            Add an Eigenmike32 at a predefined position
            >>> spa.add_microphones([("eigenmike32", [0.0, 1.0, 0.0])])

            Alternatively, the above can be written as (although this is not recommended, as it will mean only one
             "eigenmike32" can be added at any one time).
            >>> spa.add_microphones({"eigenmike32": [0.0, 1.0, 0.0]})

            Add two random microphones at two predefined positions
            >>> spa.add_microphones([[0.0, 0.0, 1.0], [-1.0, 1.0, 0.0]])

        """
        # Remove existing microphones if we wish to do this
        if not keep_existing:
            self.microphones = []
            self.ctx.clear_listeners()

        # Sanitize the microphone input into lists of [(mic_cls, mic_pos), (mic_cls, mic_pos)]
        sanitized_microphones = self._sanitize_microphone_input(microphones)
        for mic_cls, mic_pos in sanitized_microphones:
            # Try and add the microphone to the mesh
            placed = self._try_add_microphone(mic_cls, mic_pos)
            # If we can't add the microphone to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if mic_pos is None:
                    logger.warning(f"Could not place microphone in the mesh after {MAX_PLACE_ATTEMPTS} attempts. "
                                   f"Consider reducing `min_distance_from` arguments.")
                # If we were trying to add it to a specific position
                else:
                    logger.warning(f"Position {mic_pos} invalid for microphone {mic_cls.name}, "
                                   f"skipping to next microphone! Consider reducing `min_distance_from` arguments.")

        # Catch instances where no microphone exist inside the array
        if len(self.microphones) == 0:
            raise ValueError("Mesh does not contain any valid microphones!")

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
            if self._validate_source_position(point):
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
        for i, pos in enumerate(self.source_positions):
            self.ctx.add_source()
            self.ctx.set_source_position(i, pos.tolist() if isinstance(pos, np.ndarray) else pos)

    def _validate_source_position(self, pos_abs: np.ndarray) -> bool:
        """
        Validates a sound source position.
        """
        return all((
            # source must be a minimum distance from all other sources
            all(np.linalg.norm(pos_abs - pos) >= self.min_distance_from_source for pos in self.source_positions),
            # source must be a minimum distance from every microphone capsule
            #  setting axis=1 means that we calculate the distance independently for every capsule on every microphone
            all(all(np.linalg.norm(pos_abs - mic.coordinates_absolute, axis=1) >= self.min_distance_from_mic) for mic in self.microphones),
            # source must be a minimum distance from the surface
            bool(self.mesh.nearest.on_surface([pos_abs])[1][0] >= self.min_distance_from_surface),
            # source must be inside the mesh
            self._is_point_inside_mesh(pos_abs)
        ))

    @staticmethod
    def _sanitize_source_input(sources) -> list[Union[None, list]]:
        """
        Sanitizes any source input into the form [[source_coords], [source_coords]].

        `source_coords` is either a list of cartesian coordinates in XYZ format or None (in which case a random
        position will be found inside the mesh).

        Returns:
            list: the sanitized source inputs
        """

        sanitized_sources = []
        # If sources is None, we just want to add one random source
        if sources is None:
            logger.warning(f"No sources provided, placing a single source in a random position!")
            sanitized_sources.append(None)

        # If sources is an integer, we assume that this is the number of sources we want to place in random positions
        elif isinstance(sources, int):
            assert sources > 0, "Number of sources to add must be greater than 0!"
            # Iterate over the number of sources we want to try and place
            for source_idx in range(sources):
                sanitized_sources.append(None)

        # Otherwise, if sources is an iterable, we assume that this is a list of coordinates to place sources at
        elif isinstance(sources, (list, np.ndarray)):
            assert len(sources) > 0, "Number of sources to add must be greater than 0!"
            sources = utils.coerce2d(sources)
            for source in sources:
                assert len(source) == 3, "Sources must be in XYZ format!"
                sanitized_sources.append(source)

        # Otherwise, raise an error as the input is not in the expected format
        else:
            raise TypeError(f"Could not parse input with type {type(sources)}")

        return sanitized_sources

    def _try_add_source(self, position: Union[list, None], relative_mic: Optional[Type['MicArray']]) -> bool:
        """
        Try to place a source at position. Return True if successful, False otherwise.
        """
        for attempt in range(MAX_PLACE_ATTEMPTS):
            # Grab a random position for the source if required
            pos = position if position is not None else self.get_random_position()
            # If we want to express the position relative to a given microphone
            if relative_mic is not None:
                # Add the source position to the center of the microphone
                pos = relative_mic.coordinates_center + pos
            # If we have a valid position for the source
            if self._validate_source_position(pos):
                self.source_positions.append(pos)
                return True
            # If we were trying to place the source in a specific location, only make one attempt at placing it
            elif position is not None:
                break
        return False

    def add_sources(
            self,
            sources: Union[list, np.ndarray, int, None],
            keep_existing: bool = False,
            mic_idx: int = None
    ) -> None:
        """
        Add sources to the mesh at valid positions. Must be called before `simulate`.

        Arguments:
            sources (optional): the sources to be added to the mesh.
                If an integer, will be interpreted as the number of sources to add to the mesh, at random positions.
                If a list, should be a list of coordinates in XYZ form. These are assumed to be ABSOLUTE positions,
                 unless `mic_idx` is not None, in which case they are assumed to be relative to the absolute position
                 of the microphone found at `microphones[mic_idx]`
                If None, a single source will be added at a random position inside the mesh.
            keep_existing (optional): whether to remove existing sources from the mesh, defaults to removing
            mic_idx (optional): the index of the microphone that sources should be added relative to

        Examples:
            Create a space with a given mesh
            >>> spa = Space(mesh=...)

            Add a single source to the mesh with a random position
            >>> spa.add_sources()

            Add three random sources at random positions
            >>> spa.add_sources(3)

            Add one source at a specified absolute position
            >>> spa.add_sources([0.0, 0.1, 0.2])

            Add two sources at specified absolute positions
            >>> spa.add_sources([[0.0, 0.1, 0.2], [0.2, 0.1, 0.0]])

            Add one source at a specific position relative to the mic at index 1
            >>> spa.add_sources([[0.0, 0.1, 0.2]], mic_idx=1)

        """
        # Remove existing sources if we wish to do this
        if not keep_existing:
            self.source_positions = []
            self.ctx.clear_sources()

        # Sanitize the inputs into a single list
        sanitized_sources = self._sanitize_source_input(sources)
        # If we want to express our sources relative to a given microphone, grab this now
        if mic_idx is not None:
            assert len(self.microphones) >= mic_idx + 1, f"No microphone at specified index {mic_idx}!"
            desired_mic = self.microphones[mic_idx]
        else:
            desired_mic = None

        # Iterate over our sanitized source positions
        for source_pos in sanitized_sources:
            # Try and place inside the mesh: return True if placed, False if not
            placed = self._try_add_source(source_pos, desired_mic)
            # If we can't add the source to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if source_pos is None:
                    logger.warning(f"Could not place source in the mesh after {MAX_PLACE_ATTEMPTS} attempts. "
                                   f"If this is happening frequently, consider reducing the number of `sources`, "
                                   f"`min_distance_from_mic`, or `min_distance_from_source`.")
                # If we were trying to add it to a specific position
                else:
                    logger.warning(f"Position {source_pos} invalid, skipping to next source! "
                                   f"Consider reducing `min_distance_from` arguments.")

        # Sanity checking
        if len(self.source_positions) == 0:
            raise ValueError("None of the provided sources could be placed within the mesh.")

        self.source_positions = np.asarray(self.source_positions)
        self._setup_sources()

    def _simulation_sanity_check(self) -> None:
        """
        Check conditions required for simulation are met
        """
        assert len(self.source_positions) > 0, "Must have added valid sources to the mesh before calling `.simulate`!"
        assert len(self.microphones) > 0, "Must have added valid microphones to the mesh before calling `.simulate`!"
        assert all(type(m) in MICARRAY_LIST for m in self.microphones), "Non-microphone objects in microphone list"
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
        for mic_idx, mic in enumerate(self.microphones):
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
            all_irs[f"mic{str(mic_idx).zfill(3)}"] = mic.irs
        return all_irs

    def save_sofa(self, outpath: str) -> None:
        # TODO: this is almost definitely wrong/broken
        N_mics, N_sources, N_channels, N_samples = self.irs.shape

        assert N_channels == 4, "Expected 4 channels for FOA"

        M = N_mics  # Listener positions (microphones)
        R = N_sources * N_channels  # Receivers = sources * FOA channels
        C = 3
        E, I = 1, 1

        # Check source positions shape (should be N_sources x 3)
        assert self.source_positions.shape == (N_sources, C)

        if os.path.exists(outpath):
            logger.warning(f"File {outpath} already exists, overwriting!")

        rootgrp = Dataset(outpath, "w", format="NETCDF4")

        rootgrp.Conventions = "SOFA"
        rootgrp.SOFAConventions = "SingleRoomSRIR"
        rootgrp.SOFAConventionsVersion = "1.0"
        rootgrp.APIName = "pysofaconventions"
        rootgrp.APIVersion = "0.1.5"
        rootgrp.DataType = "FIR"
        current_time = time.ctime(time.time())
        rootgrp.DateCreated = current_time
        rootgrp.DateModified = current_time

        for dim_str, dim_val in zip(["M", "R", "N", "E", "I", "C"], [M, R, N_samples, E, I, C]):
            rootgrp.createDimension(dim_str, dim_val)

        # ListenerPosition (M, C): mic positions
        listener_pos_var = rootgrp.createVariable("ListenerPosition", "f8", ("M", "C"))
        listener_pos_var.Units = "metre"
        listener_pos_var.Type = "cartesian"
        listener_pos_var[:] = self.microphones  # shape (M, 3)

        # ListenerUp (I, C)
        listener_up_var = rootgrp.createVariable("ListenerUp", "f8", ("I", "C"))
        listener_up_var.Units = "metre"
        listener_up_var.Type = "cartesian"
        listener_up_var[:] = np.asarray([[0, 0, 1]])

        # ListenerView (I, C)
        listener_view_var = rootgrp.createVariable("ListenerView", "f8", ("I", "C"))
        listener_view_var.Units = "metre"
        listener_view_var.Type = "cartesian"
        listener_view_var[:] = np.asarray([[1, 0, 0]])

        # EmitterPosition (E, C, I)
        emitter_position_var = rootgrp.createVariable("EmitterPosition", "f8", ("E", "C", "I"))
        emitter_position_var.Units = "metre"
        emitter_position_var.Type = "spherical"
        emitter_position_var[:] = np.zeros((E, C, I))

        # SourcePosition (R, C, I): for each receiver, need a position
        # Since receivers = sources * channels, we expand sources accordingly

        source_positions_expanded = np.repeat(self.source_positions, N_channels, axis=0)  # shape (R, 3)

        source_position_var = rootgrp.createVariable("SourcePosition", "f8", ("R", "C"))
        source_position_var.Units = "metre"
        source_position_var.Type = "cartesian"
        source_position_var[:] = source_positions_expanded

        # SourceUp (I, C)
        source_up_var = rootgrp.createVariable("SourceUp", "f8", ("I", "C"))
        source_up_var.Units = "metre"
        source_up_var.Type = "cartesian"
        source_up_var[:] = np.asarray([[0, 0, 1]])

        # SourceView (I, C)
        source_view_var = rootgrp.createVariable("SourceView", "f8", ("I", "C"))
        source_view_var.Units = "metre"
        source_view_var.Type = "cartesian"
        source_view_var[:] = np.asarray([[1, 0, 0]])

        # ReceiverPosition (R, C, I)
        receiver_position_var = rootgrp.createVariable("ReceiverPosition", "f8", ("R", "C", "I"))
        receiver_position_var.Units = "metre"
        receiver_position_var.Type = "cartesian"

        # Each receiver corresponds to a (source, channel) pair
        # Receiver positions are mic positions repeated for each source * channel
        receiver_positions = np.repeat(self.microphones, N_sources * N_channels, axis=0)[:R]
        receiver_position_var[:] = receiver_positions[:, :, np.newaxis]  # shape (R, 3, 1)

        # Sampling rate (I,)
        sampling_rate_var = rootgrp.createVariable("Data.SamplingRate", "f8", ("I",))
        sampling_rate_var.Units = "hertz"
        sampling_rate_var[:] = utils.SAMPLE_RATE

        # Delay (I, R)
        delay_var = rootgrp.createVariable("Data.Delay", "f8", ("I", "R"))
        delay_var[:, :] = np.zeros((I, R))

        # Data.IR (M, R, N)
        data_ir_var = rootgrp.createVariable("Data.IR", "f8", ("M", "R", "N"))
        data_ir_var.ChannelOrdering = "acn"
        data_ir_var.Normalization = "sn3d"

        # Rearrange rirs to (M, R, N)
        # rirs shape: (N_mics, N_sources, N_channels, N_samples)
        # Want: M=N_mics, R=N_sources * N_channels, N=N_samples

        rirs_transposed = self.irs.transpose(0, 1, 2, 3)  # (M, N_sources, N_channels, N_samples)
        rirs_reshaped = rirs_transposed.reshape(M, R, N_samples)
        data_ir_var[:] = rirs_reshaped

        rootgrp.close()
        logger.info(f"SOFA file saved to {outpath}")

    def create_scene(self, mic_radius: float = 0.2, source_radius: float = 0.1) -> trimesh.Scene:
        """
        Creates a trimesh.Scene with the Space's mesh, microphone position, and sources all added

        Returns:
            trimesh.Scene: The rendered scene, that can be shown in e.g. a notebook with the `.show()` command
        """
        scene = self.mesh.scene()
        # This just adds the microphone positions
        for mic in self.microphones:
            for capsule in mic.coordinates_absolute:
                add_sphere(scene, capsule, color=[255, 0, 0], r=mic_radius)
        # This adds the sound sources, with different color + radius
        for source in self.source_positions:
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
        mic_positions = np.vstack([m.coordinates_absolute for m in self.microphones])
        for ax_, idx, color, ylab, title in zip(ax.flatten(), [1, 2], ["red", "blue"], ["Y", "Z"], ["Top", "Side"]):
            # Scatter the vertices first
            ax_.scatter(vertices[:, 0], vertices[:, idx], c='gray', alpha=0.1, s=1)
            # Then the microphone and source positions
            ax_.scatter(mic_positions[:, 0], mic_positions[:, idx], c='red', s=100, label='Microphone')
            ax_.scatter(
                self.source_positions[:, 0], self.source_positions[:, idx], c='blue', s=25, alpha=0.5, label='Sources'
            )
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
        for mic_idx, mic in enumerate(self.microphones):
            mic_idx = str(mic_idx).zfill(3)
            for caps_idx, caps in enumerate(mic.irs):
                caps_idx = str(caps_idx).zfill(3)
                for source_idx, source in enumerate(caps):
                    source_idx = str(source_idx).zfill(3)
                    fname = os.path.join(outdir, f"mic{mic_idx}_capsule{caps_idx}_source{source_idx}.wav")
                    wavfile.write(fname, utils.SAMPLE_RATE, source)


if __name__ == "__main__":
    import glob

    # For reproducible random source placement
    utils.seed_everything(utils.SEED)
    # Get all the object files we're considering here
    dataset_dir = os.path.join(utils.get_project_root(), "resources/meshes")
    mesh_paths = [glob.glob(e, root_dir=dataset_dir) for e in ['**/*.glb', '**/*.obj']]
    # Flatten and combine with the root directory
    mesh_paths = [os.path.join(dataset_dir, x) for xs in mesh_paths for x in xs]
    # Iterate over all the .glb files inside the meshes directory
    for mesh_path in mesh_paths:
        logger.info(f"Processing {mesh_path}")
        # Create the space and add N random sources
        room = Space(mesh=mesh_path)
        room.add_microphones(microphones=["ambeovr", "ambeovr", "ambeovr"])
        room.add_sources(5)
        # Simulate the room impulse responses
        room.simulate()
        # Visualize microphone and sources inside the scene
        room_fig = room.create_plot()
        # Dump the plot to an image and the RIRs to WAV files
        basename = os.path.dirname(mesh_path)
        room_fig.savefig(os.path.join(basename, "out.png"))
        room.save_irs_to_wav(basename)