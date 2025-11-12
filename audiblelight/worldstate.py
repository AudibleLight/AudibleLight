#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for representing triangular meshes, handling spatial operations, generating RIRs."""

import json
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any, Optional, Type, TypeVar, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from deepdiff import DeepDiff
from loguru import logger
from pysofaconventions import SOFAFile
from rlr_audio_propagation import Config, Context
from scipy.spatial import KDTree
from tqdm import tqdm

from audiblelight import config, custom_types, utils
from audiblelight.micarrays import (
    CHANNEL_LAYOUT_TYPES,
    MICARRAY_LIST,
    MicArray,
    dynamically_define_micarray,
    sanitize_microphone_input,
)

FACE_FILL_COLOR = [255, 0, 0, 255]
MATERIALS_JSON = str(
    utils.sanitise_filepath(
        utils.get_project_root() / "resources/mp3d_material_config.json"
    )
)

# These are the only moving event trajectores that are valid
VALID_MOVING_EVENT_TRAJECTORIES = [
    "linear",
    "semicircular",
    "sine",
    "sawtooth",
    "random",
]


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
    if loaded_mesh.units != config.MESH_UNITS:
        logger.warning(
            f"Mesh {mesh_fpath.stem} has units {loaded_mesh.units}, converting to {config.MESH_UNITS}"
        )
        loaded_mesh = loaded_mesh.convert_units(config.MESH_UNITS, guess=True)
    return loaded_mesh


def get_broken_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Get the idxs of broken faces in a mesh. Uses copies to prevent anything being set inplace.
    """
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
    scene: trimesh.Scene,
    pos: np.ndarray,
    color: Optional[list[int]] = None,
    r: Optional[custom_types.Numeric] = 0.2,
) -> None:
    """
    Adds a sphere object to a scene with given position, color, and radius
    """
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

    def __init__(
        self, alias: str, coordinates_absolute: np.ndarray, sofa_idx: int = None
    ):
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

        # Index of the IR/position within the SOFA file
        self.sofa_idx = (
            utils.sanitise_positive_number(sofa_idx, cast_to=int)
            if sofa_idx is not None
            else None
        )

        self.has_direct_paths: OrderedDict[str, bool] = OrderedDict()

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
                    if all([isinstance(em, Emitter) for em in obj]):
                        coords = np.vstack([em.coordinates_absolute for em in obj])

                    else:
                        raise TypeError(
                            "Cannot handle input with type {}".format(type(obj))
                        )

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
        # Create output dictionary
        out_dict = dict(
            alias=self.alias,
            coordinates_absolute=utils.coerce_nested_inputs(self.coordinates_absolute),
            has_direct_paths=self.has_direct_paths,
        )
        if self.sofa_idx:
            out_dict["sofa_idx"] = self.sofa_idx
        return out_dict

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
            # "coordinates_relative_cartesian",
            # "coordinates_relative_polar",
        ]:
            if k not in copied_dict:
                raise KeyError(f"Missing key '{k}'")

            # Need to convert lists back to arrays
            copied_dict[k] = unserialise(copied_dict[k])

        # Instantiate the class with the correct alias and absolute coordinates
        kws = dict(
            alias=copied_dict["alias"],
            coordinates_absolute=copied_dict["coordinates_absolute"],
        )

        # Add in sofa IDX if required and present
        if "sofa_idx" in copied_dict.keys():
            kws["sofa_idx"] = copied_dict["sofa_idx"]

        return cls(**kws)


class WorldState:
    """
    Represents a 3D space defined by a room, microphone position(s), and emitter position(s).

    Should not be used directly: instead, a child class (e.g., `WorldStateRIR`, `WorldStateSOFA`) should be used.
    """

    name = "_default"

    def __init__(self):
        # Store emitter and mic positions in here to access later; these should be in ABSOLUTE form
        self.emitters = OrderedDict()
        self.microphones = OrderedDict()
        self._irs = None  # will be updated when calling `simulate`

        # These need to be defined for typehinting
        #  They are overridden in the child classes
        self.mesh = None
        self.waypoints = None
        self.ctx = None

    def _update(self) -> None:
        """
        Updates the state, setting emitter positions correctly.
        """
        raise NotImplementedError

    def simulate(self) -> None:
        """
        Simulates audio propagation in the state with the current listener and sound emitter positions.
        """
        raise NotImplementedError

    def get_valid_position(self) -> np.ndarray:
        """
        Get a valid position to place an object inside the state

        Returns:
             np.ndarray: the random position to place an object inside the mesh
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate a `WorldState` from a dictionary.
        """
        if "backend" not in input_dict.keys():
            raise KeyError("Must set 'backend' key to parse from dictionary")

        # Get the desired backend and instantiate using the child class function
        desired_backend = get_worldstate_from_string(input_dict["backend"])
        return desired_backend.from_dict(input_dict)

    @property
    def irs(self) -> OrderedDict[str, np.ndarray]:
        """
        Returns a dictionary of IRs in the shape {mic000: (N_capsules, N_emitters, N_samples), mic001: (...)}
        """
        if self._irs is None:
            raise AttributeError(
                "IRs have not been simulated yet: add microphones and emitters and call `simulate`."
            )
        else:
            return self._irs

    def get_irs(self) -> OrderedDict[str, np.ndarray]:
        """
        Get the IRs from the state
        """
        raise NotImplementedError

    @property
    def num_emitters(self) -> int:
        """
        Returns the number of emitters in the state.

        Note that this is not the same as calling `len(self.emitters)`: the total number of emitters is equivalent
        to the length of ALL lists inside this dictionary
        """
        return sum(len(v) for v in self.emitters.values())

    def __len__(self) -> int:
        """
        Returns the number of objects in the mesh (i.e., number of microphones + emitters)
        """
        return len(self.microphones) + self.num_emitters

    def __str__(self) -> str:
        """
        Returns a string representation of the WorldState
        """
        return (
            f"'{self.__class__.__name__}' with {len(self)} objects "
            f"({len(self.microphones)} microphones, {self.num_emitters} emitters)"
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

        # Objects with a different type objects are always not equal
        if not isinstance(other, type(self)):
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

    def get_emitter(self, alias: str, emitter_idx: Optional[int] = 0) -> Emitter:
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

    def add_microphone(
        self,
        microphone_type: Optional[Union[str, Type["MicArray"]]] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        alias: Optional[str] = None,
        keep_existing: Optional[bool] = True,
    ) -> None:
        """
        Add a microphone to the space.
        """
        raise NotImplementedError

    def add_microphones(
        self,
        microphone_types: Optional[list[Union[str, Type["MicArray"]]]] = None,
        positions: Optional[list[Union[list, np.ndarray]]] = None,
        aliases: Optional[list[str]] = None,
        keep_existing: Optional[bool] = True,
        raise_on_error: Optional[bool] = True,
    ) -> None:
        """
        Add multiple microphones to the state.
        """
        raise NotImplementedError

    def add_emitter(
        self,
        position: Optional[Union[list, np.ndarray]] = None,
        alias: Optional[str] = None,
        mic: Optional[str] = None,
        keep_existing: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
    ) -> None:
        """
        Add an emitter to the state.
        """
        raise NotImplementedError

    def add_emitters(
        self,
        positions: Optional[Union[list, np.ndarray]] = None,
        aliases: Optional[list[str]] = None,
        mics: Optional[Union[list[str], str]] = None,
        n_emitters: Optional[int] = None,
        keep_existing: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        raise_on_error: Optional[bool] = True,
    ) -> None:
        """
        Add emitters to the state.
        """
        raise NotImplementedError

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
        max_place_attempts: Optional[int] = config.MAX_PLACE_ATTEMPTS,
    ) -> None:
        """
        Add both a microphone and emitter with specified relationship.
        """
        raise NotImplementedError

    def _validate_position(self, pos_abs: np.ndarray) -> bool:
        """
        Validates a position or array of positions with respect to the state and objects inside it.
        """
        raise NotImplementedError

    def define_trajectory(
        self,
        duration: custom_types.Numeric,
        starting_position: Optional[Union[np.ndarray, list]] = None,
        velocity: Optional[custom_types.Numeric] = config.DEFAULT_EVENT_VELOCITY,
        resolution: Optional[custom_types.Numeric] = config.DEFAULT_EVENT_RESOLUTION,
        shape: Optional[str] = None,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
    ) -> np.ndarray:
        """
        Defines a trajectory for a moving sound event with specified spatial bounds and event duration.
        """
        raise NotImplementedError

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

    def clear_microphones(self) -> None:
        """
        Removes all current microphones.
        """
        self.microphones = OrderedDict()
        # Always update the state after clearing, regardless of `add_to_state` setting
        self._update()

    def clear_emitters(self) -> None:
        """
        Removes all current emitters.
        """
        self.emitters = OrderedDict()
        # Always update the state after clearing, regardless of `add_to_state` setting
        self._update()

    def clear_microphone(self, alias: str) -> None:
        """
        Given an alias for a microphone, clear that microphone if it exists and update the state.
        """
        if alias in self.microphones.keys():
            del self.microphones[alias]
            # Always update the state after clearing, regardless of `add_to_state` setting
            self._update()
        else:
            raise KeyError("Microphone alias '{}' not found.".format(alias))

    def clear_emitter(self, alias: str) -> None:
        """
        Given an alias for an emitter, clear that emitter and update the state.
        """
        if alias in self.emitters.keys():
            del self.emitters[alias]
            # Always update the state after clearing, regardless of `add_to_state` setting
            self._update()
        else:
            raise KeyError("Emitter alias '{}' not found.".format(alias))

    def _parse_valid_microphone_aliases(
        self, aliases: Optional[Union[bool, list, str]]
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

    def path_exists_between_points(
        self, point_a: np.ndarray, point_b: np.ndarray
    ) -> bool:
        """
        Returns True if a direct point exists between point_a and point_b in the mesh, False otherwise.
        """
        raise NotImplementedError

    def _add_emitters_without_validating(
        self,
        emitters: Union[list, np.ndarray],
        alias: Optional[str],
    ) -> None:
        """
        Adds emitters from a list **without checking** that they are valid.
        """
        raise NotImplementedError


class WorldStateRLR(WorldState):
    """
    A WorldState where audio propagation is simulated inside a 3D-scanned mesh using acoustic ray-tracing.

    Attributes:
        mesh (str, Path): The path to the mesh on the disk.
        microphones (np.array): Position of the microphone in the mesh.
        ctx (rlr_audio_propagation.Context): The context for audio propagation simulation.
        emitters (np.array): relative positions of sound emitter
    """

    name = "RLR"

    def __init__(
        self,
        mesh: Union[str, Path],
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        empty_space_around_mic: Optional[
            custom_types.Numeric
        ] = config.EMPTY_SPACE_AROUND_MIC,
        empty_space_around_emitter: Optional[
            custom_types.Numeric
        ] = config.EMPTY_SPACE_AROUND_EMITTER,
        empty_space_around_surface: Optional[
            custom_types.Numeric
        ] = config.EMPTY_SPACE_AROUND_SURFACE,
        empty_space_around_capsule: Optional[
            custom_types.Numeric
        ] = config.EMPTY_SPACE_AROUND_CAPSULE,
        add_to_context: Optional[bool] = True,
        ensure_minimum_weighted_average_ray_length: Optional[bool] = False,
        minimum_weighted_average_ray_length: Optional[
            custom_types.Numeric
        ] = config.MIN_AVG_RAY_LENGTH,
        repair_threshold: Optional[custom_types.Numeric] = None,
        waypoints_json: Optional[Union[str, Path]] = None,
        material: Optional[str] = None,
        rlr_kwargs: Optional[dict] = None,
    ):
        """
        Initializes the WorldState with a mesh and sets up the audio context.

        Args:
            mesh (str|Path): The name of the mesh file. Units will be coerced to meters when loading
            sample_rate (Numeric): the sample rate to use in the RLR engine
            empty_space_around_mic (float): minimum meters new emitters/mics will be placed from center of other mics
            empty_space_around_emitter (float): minimum meters new emitters/mics will be placed from other emitters
            empty_space_around_surface (float): minimum meters new emitters/mics will be placed from mesh emitters
            empty_space_around_capsule (float): minimum meters new emitters/mics will be placed from mic capsules
            add_to_context (bool): if False, the ray-tracing context will ONLY be updated when running
                `WorldState.simulate`. This is ideal in large-scale data generation pipelines. If True, the state
                will be updated every time a new Microphone or Emitter is added. This is ideal for interactive use.
            ensure_minimum_weighted_average_ray_length (bool): if True, random points can only be sampled from within
                the mesh when they have a weighted average ray length of at least `minimum_weighted_average_ray_length`
            minimum_weighted_average_ray_length (float): value to consider when locating points in the mesh; only
                evaluated when `ensure_minimum_weighted_average_ray_length` is True
            repair_threshold (float, optional): when the proportion of broken faces on the mesh is below this value,
                repair the mesh and fill holes. If None, will never repair the mesh.
            waypoints_json (str|Path, optional): path pointing towards a JSON list containing waypoints for this mesh.
                If not provided, will attempt to infer from the filename and set to None if cannot be found.
            material (str): the name of a material to use, defaults to None (i.e., Default material)
            rlr_kwargs (dict, optional): additional keyword arguments to pass to the RLR audio propagation library.
        """
        # This initialises microphones, emitters dictionaries
        super().__init__()

        self.add_to_state = add_to_context

        self.sample_rate = utils.sanitise_positive_number(sample_rate, cast_to=int)

        # Distances from objects/mesh surfaces
        self.empty_space_around_mic = utils.sanitise_positive_number(
            empty_space_around_mic
        )
        self.empty_space_around_surface = utils.sanitise_positive_number(
            empty_space_around_surface
        )
        self.empty_space_around_emitter = utils.sanitise_positive_number(
            empty_space_around_emitter
        )
        self.empty_space_around_capsule = utils.sanitise_positive_number(
            empty_space_around_capsule
        )

        # Checking minimum weighted average ray length
        self.ensure_minimum_weighted_average_ray_length = (
            ensure_minimum_weighted_average_ray_length
        )
        self.minimum_weighted_average_ray_length = utils.sanitise_positive_number(
            minimum_weighted_average_ray_length
        )

        # Load in the trimesh object
        self.mesh = load_mesh(mesh)

        # Try and load up waypoints for this mesh
        self.waypoints = self.load_mesh_navigation_waypoints(waypoints_json)

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

        # Setting up audio configuration
        self.material = self._validate_material(material)
        self.cfg = self._parse_rlr_config(rlr_kwargs)
        self.ctx = None
        # If required, create the audio context now
        if self.add_to_state:
            self._setup_audio_context()

    # noinspection PyUnreachableCode
    def _update(self) -> None:
        """
        Updates the state, setting emitter positions and adding all items to the ray-tracing context correctly.
        """
        # (re)make the audio context
        self._setup_audio_context()

        # Update the ray-tracing listeners
        if len(self.microphones) > 0:
            counter = 0
            # Iterate through all the microphones we've added
            for mic in self.microphones.values():
                # Iterate through all capsules of this micarray class
                for caps in mic.coordinates_absolute:
                    # Add the listener in with the correct layout for the current mic
                    self.ctx.add_listener(mic.channel_layout)
                    # Set the position of the listener and update the counter
                    self.ctx.set_listener_position(
                        counter,
                        caps.tolist() if isinstance(caps, np.ndarray) else caps,
                    )
                    counter += 1

        # Update the ray-tracing sources
        if self.num_emitters > 0:
            emitter_counter = 0
            for emitter_alias, emitter_list in self.emitters.items():
                for emitter in emitter_list:
                    # Update the coordinates of the emitter WRT other microphones, emitters
                    # emitter.update_coordinates(self.emitters)
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

                    # Add in a flag if the Emitter has a direct path to each microphone
                    for mic_alias, mic in self.microphones.items():
                        emitter.has_direct_paths[mic_alias] = (
                            self.path_exists_between_points(
                                mic.coordinates_center, emitter.coordinates_absolute
                            )
                        )

    def _parse_rlr_config(self, rlr_kwargs: dict) -> Config:
        """
        Parses the configuration for the ray-tracing engine
        """
        # Create the configuration object with the default settings
        cfg = Config()
        if rlr_kwargs is None:
            rlr_kwargs = {}

        # Ensure sample rate always present  in config
        if "sample_rate" not in rlr_kwargs.keys():
            rlr_kwargs["sample_rate"] = self.sample_rate

        # Iterate over our passed parameters and update as required
        for rlr_kwarg, rlr_val in rlr_kwargs.items():

            # Ensure sample rate matches the one passed in to the worldstate
            if rlr_kwarg == "sample_rate":
                if rlr_val != self.sample_rate:
                    raise ValueError(
                        f"Mismatching sample rate (expected {self.sample_rate}, got {rlr_val})"
                    )

            # Set attribute correctly in the ray tracing engine
            if hasattr(cfg, rlr_kwarg):
                setattr(cfg, rlr_kwarg, rlr_val)
            else:
                raise AttributeError(f"Ray-tracing engine has no attribute {rlr_kwarg}")

        return cfg

    def calculate_weighted_average_ray_length(
        self,
        point: np.ndarray,
        num_rays: Optional[custom_types.Numeric] = config.NUM_RAYS,
    ) -> custom_types.Numeric:
        """
        Estimate how spatially "open" a point is by computing the weighted average length of rays cast from that point.

        Rays are emitted uniformly from the given point across 3D space. Each ray is traced until it intersects with
        the mesh surface. The distances of these intersections are squared and used as weights to calculate a weighted
        average, emphasising longer, unobstructed paths. This can be used as a heuristic for how suitable a point is
        within a mesh (e.g., avoiding corners)

        If any rays fail to intersect the mesh (due to holes or open surfaces), those rays are ignored, and a warning
        is logged.

        Arguments:
            point (np.ndarray): a 3D coordinate (shape: (3,)) representing the origin of the rays.
            num_rays (int): number of random rays to cast from the point (default is 100).

        Returns:
            float: The weighted average distance of ray intersections: higher values indicate more open surroundings
        """
        # Sanitisation of inputs
        num_rays = utils.sanitise_positive_number(num_rays, cast_to=int)
        point = utils.sanitise_coordinates(point)

        # Generate random azimuthal angles for each ray
        angles = np.random.uniform(0, 2 * np.pi, num_rays)
        # Generate random elevation angles for each ray
        elevations = np.random.uniform(-np.pi / 2, np.pi / 2, num_rays)

        # Convert spherical coordinates (angles, elevations) to  Cartesian 3D direction vectors
        cos_elevation = np.cos(elevations)
        directions = np.empty((num_rays, 3))
        directions[:, 0] = cos_elevation * np.cos(angles)  # x
        directions[:, 1] = cos_elevation * np.sin(angles)  # y
        directions[:, 2] = np.sin(elevations)  # z

        # Repeat the origin point for each ray so rays start from the same position
        origins = np.broadcast_to(point, (num_rays, 3))
        # Cast rays from the origin in the computed directions and find the longest intersection distances with the mesh
        distances = trimesh.proximity.longest_ray(self.mesh, origins, directions)

        # We can get `inf` values here, likely due to holes in the mesh causing a ray never to intersect
        if np.isinf(distances).any():
            # For simplicity, we can just remove these here but raise a warning
            logger.warning(
                f"Some rays cast from point {point} have infinite distances: is the mesh watertight?"
            )
            distances = distances[distances != np.inf]

        # Compute weights by squaring the distances to give more importance to longer rays
        weights = distances**2
        # Calculate weighted average of the distances using the computed weights and return
        return np.sum(distances * weights) / np.sum(weights)

    @staticmethod
    def _validate_material(material: Optional[str] = None) -> str:
        """
        Validates that an input material is acceptable for the ray-tracing engine.

        Arguments:
            material (str): name of the material to use. Must be a key inside `../references/mp3d_material_config.json`.
                Defaults to "Default" material if not provided.

        Returns:
            str: name of the material to use in the ray-tracing engine.
        """
        with open(MATERIALS_JSON, "r") as js_in:
            js_out = json.load(js_in)
        valid_materials = {mat["name"] for mat in js_out["materials"]}

        if not material:
            material = "Default"

        if material not in valid_materials:
            raise ValueError(f"Material {material} is not a valid material.")

        return material

    def _setup_audio_context(self) -> None:
        """
        Initializes the audio context and configures the mesh for the context.
        """

        def get_audio(_) -> None:
            raise NotImplementedError(
                "Do not call `WorldState.ctx.get_audio` directly: instead, use `WorldState.get_irs`."
            )

        # If the context doesn't exist, make it
        if self.ctx is None:
            self.ctx = Context(self.cfg)
        # If the context does exist, reset it
        else:
            self.ctx.reset(self.cfg)
            if self.ctx.get_listener_count() > 0:
                self.ctx.clear_listeners()
            if self.ctx.get_source_count() > 0:
                self.ctx.clear_sources()

        self.ctx.set_material_database_json(MATERIALS_JSON)

        # Add the mesh into the context
        self.ctx.add_object()
        self.ctx.set_object_position(0, [0, 0, 0])
        self.ctx.add_mesh_vertices(self.mesh.vertices.flatten().tolist())
        self.ctx.add_mesh_indices(self.mesh.faces.flatten().tolist(), 3, self.material)
        self.ctx.finalize_object_mesh(0)

        # Need to monkey-patch get_audio for Context obj as it won't work with multiple channel layout types
        self.ctx.get_audio = MethodType(get_audio, self.ctx)

    def _try_add_microphone(
        self, mic_cls, position: Optional[Union[list, np.ndarray]], alias: str
    ) -> bool:
        """
        Try to place a microphone of type mic_cls at position with given alias. Return True if successful.
        """
        if alias in self.microphones.keys():
            raise KeyError(f"Alias {alias} already exists in microphone dictionary")

        for attempt in range(config.MAX_PLACE_ATTEMPTS):
            # Grab a random position for the microphone if required
            pos = position if position is not None else self.get_valid_position()
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

    def add_microphone(
        self,
        microphone_type: Optional[Union[str, Type["MicArray"]]] = None,
        position: Optional[Union[list, np.ndarray]] = None,
        alias: Optional[str] = None,
        keep_existing: Optional[bool] = True,
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
            >>> spa = WorldStateRLR(mesh=...)

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
                    f"Could not place microphone in the mesh after {config.MAX_PLACE_ATTEMPTS} attempts. "
                    f"Consider reducing `empty_space_around` arguments."
                )
            # If we were trying to add it to a specific position
            else:
                raise ValueError(
                    f"Position {position} invalid for microphone {sanitized_microphone.name}. "
                    f"Consider reducing `empty_space_around` arguments."
                )

        # If placed successfully, update the state
        else:
            if self.add_to_state:
                self._update()

    def add_microphones(
        self,
        microphone_types: Optional[list[Union[str, Type["MicArray"]]]] = None,
        positions: Optional[list[Union[list, np.ndarray]]] = None,
        aliases: Optional[list[str]] = None,
        keep_existing: Optional[bool] = True,
        raise_on_error: Optional[bool] = True,
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
            >>> spa = WorldStateRLR(mesh=...)

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
                        f"Could not place microphone in the mesh after {config.MAX_PLACE_ATTEMPTS} attempts. "
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

        # Update the state after placing everything
        if self.add_to_state:
            self._update()

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
        max_place_attempts: Optional[int] = config.MAX_PLACE_ATTEMPTS,
    ) -> None:
        """
        Add both a microphone and emitter with specified relationship.

        The microphone will be placed in a random, valid position. The emitter will then be placed relative to the
        microphone, either in Cartesian or spherical coordinates.

        Args:
            position (np.ndarray): Array of form [X, Y, Z]
            polar: whether the coordinates are provided in spherical form. If True:
                - Azimuth (X) must be between 0 and 360
                - Elevation (Y) must be between -90 and 90
                - Radius (Z) must be a positive value, measured in the same units given by the mesh.
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
            >>> spa = WorldStateRLR(mesh=...)

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
        #  returns a 2D array, we just want 1D
        if polar:
            emitter_offset = utils.polar_to_cartesian(emitter_offset)[0]

        # Attempt to find valid positions for both microphone and emitter
        for attempt in range(max_place_attempts):
            # Get a random position for the microphone
            mic_pos = self.get_valid_position()

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

                # Update the state and return
                if self.add_to_state:
                    self._update()
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

    def get_valid_position(self) -> np.ndarray:
        """
        Get a valid position to place an object inside the state

        If `ensure_minimum_weighted_average_ray_length` is enabled, this function attempts to find a position that
        meets the minimum openness criteria, measured by the weighted average ray length from the candidate point. It
        will try up to `MAX_PLACE_ATTEMPTS` times before returning the last attempted position.

        Returns:
             np.ndarray: the random position to place an object inside the mesh
        """
        # Sample an initial position inside the mesh
        #  If we don't care about minimum weighted average ray length, we'll just use this
        mic_pos = self.get_random_point_inside_mesh()

        # If we care about checking vs minimum weighted average ray length, we need to iterate
        if self.ensure_minimum_weighted_average_ray_length:
            for attempt in range(config.MAX_PLACE_ATTEMPTS):

                # Compute the weighted average ray length with this position and return if the position is acceptable
                if (
                    self.calculate_weighted_average_ray_length(mic_pos)
                    >= self.minimum_weighted_average_ray_length
                ):
                    logger.info(f"Found suitable position after {attempt + 1} attempts")
                    return mic_pos

                # Generate a new position
                else:
                    mic_pos = self.get_random_point_inside_mesh()

            # If we haven't found an acceptable position, log this and use the most recent one.
            logger.error(
                f"Could not find a suitable position after {config.MAX_PLACE_ATTEMPTS} attempts. "
                f"Using the last attempted position: {mic_pos}."
            )

        return mic_pos

    def get_random_point_inside_mesh(
        self, batch_size: Optional[custom_types.Numeric] = config.POINT_BATCH_SIZE
    ) -> np.ndarray:
        """
        Generates a random valid point inside the mesh.

        N positions will be generated in batches and a whole batch will be checked at once to take advantage of
        `numpy` vectorisation. In initial experiments, using a batch size of 10 resulted in a speed-up of 1.5x over
        a batch size of 1 (i.e., no batching). Improvements level off and decrease after, with a batch size of 100
        resulting in 2x worse performance than no batching.

        Arguments:
            batch_size (int): number of points to generate in a single batch, defaults to 10

        Returns:
            np.array: A valid point within the mesh in XYZ format
        """
        min_bound, max_bound = self.mesh.bounds
        # Keep iterating until we get at least one valid point
        while True:
            points = np.random.uniform(min_bound, max_bound, size=(batch_size, 3))
            # Returns a boolean mask of shape (batch_size,)
            mask = self._get_valid_positions_mask(points)
            # Index and get a random element from the valid items in the batch
            if np.any(mask):
                valids = np.flatnonzero(mask)
                return points[np.random.choice(valids)]

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

        Returns:
             bool: True if valid, False if not. If multiple arrays provided, return True only if all are valid.
        """
        # Create the mask: one element per coordinate
        mask = self._get_valid_positions_mask(pos_abs)
        # If the position is valid, all elements should be True
        return bool(mask.all())

    def _get_valid_positions_mask(self, pos_abs: np.ndarray) -> np.ndarray:
        """
        Validates an array of positions with respect to the mesh and objects inside it.

        Returns:
            np.ndarray: a boolean mask of shape (N,) where True = valid, False = invalid.
        """
        positions = utils.coerce2d(pos_abs)
        if positions.shape[1] != 3:
            raise ValueError("Expected input to have shape (N, 3) for XYZ coordinates")

        # Create the empty mask with the same length as the input
        num_positions = positions.shape[0]
        valid_mask = np.ones(num_positions, dtype=bool)

        # Distance from emitters
        if self.emitters:
            emitter_coords = np.vstack(
                [
                    emitter.coordinates_absolute
                    for emitter_list in self.emitters.values()
                    for emitter in emitter_list
                ]
            )
            emitter_dists = np.linalg.norm(
                positions[:, None, :] - emitter_coords[None, :, :], axis=2
            )
            too_close_to_emitter = np.any(
                emitter_dists < self.empty_space_around_emitter, axis=1
            )
            valid_mask &= ~too_close_to_emitter

        # Distance from microphones
        if self.microphones:
            for attr, thresh in zip(
                ["coordinates_center", "coordinates_absolute"],
                [self.empty_space_around_mic, self.empty_space_around_capsule],
            ):
                mic_coords = np.vstack(
                    [getattr(mic, attr) for mic in self.microphones.values()]
                )
                mic_dists = np.linalg.norm(
                    positions[:, None, :] - mic_coords[None, :, :], axis=2
                )
                too_close_to_mic = np.any(mic_dists < thresh, axis=1)
                valid_mask &= ~too_close_to_mic

        # Distance from mesh surface
        surface_dists = self.mesh.nearest.on_surface(positions)[1]
        too_close_to_surface = surface_dists < self.empty_space_around_surface
        valid_mask &= ~too_close_to_surface

        # Inside mesh check
        inside_mask = np.array([self._is_point_inside_mesh(p) for p in positions])
        valid_mask &= inside_mask

        return valid_mask

    def _try_add_emitter(
        self,
        position: Optional[list],
        relative_mic: Optional[Type["MicArray"]],
        alias: str,
        path_between: list[str],
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
    ) -> bool:
        """
        Attempt to add a emitter at the given position with the specified alias.
        Returns True if placement is successful, otherwise False.
        """
        # True if we want a specific position, False if not
        position_is_assigned = position is not None
        # If we have already provided a position, this loop will only iterate once
        #  Otherwise, we want a random position, so we iterate N times until the position is valid
        for attempt in range(1 if position_is_assigned else max_place_attempts):
            # Get a random position if required or use the assigned one
            pos = position if position_is_assigned else self.get_valid_position()
            if len(pos) != 3:
                raise ValueError(f"Expected three coordinates but got {len(pos)}")
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
            emitter = Emitter(
                alias=alias, coordinates_absolute=utils.sanitise_coordinates(pos)
            )
            # Add the emitter to the list created for this alias, or create the list if it doesn't exist
            if alias in self.emitters:
                self.emitters[alias].append(emitter)
            else:
                self.emitters[alias] = [emitter]
            return True
        # Cannot place: return False
        return False

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

    def add_emitter(
        self,
        position: Optional[Union[list, np.ndarray]] = None,
        alias: Optional[str] = None,
        mic: Optional[str] = None,
        keep_existing: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
    ) -> None:
        """
        Add an emitter to the state.

        If `position` is provided, it must be in absolute (cartesian) terms.
        If `mic` is a key inside `microphones`, `position` is assumed to be relative to that microphone.

        Arguments:
            position: Location to add the emitter, defaults to a random, valid location.
            alias: String reference to access the emitter inside the `self.emitters` dictionary.
            mic: String reference to a microphone inside `self.microphones`;
                when provided, `position` is interpreted as RELATIVE to the center of this microphone
            keep_existing (optional): Whether to keep existing emitters from the mesh or remove, defaults to keep
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for an emitter.
            max_place_attempts (Numeric): the number of times to try and create the trajectory.

        Examples:
            Create a state with a given mesh and add a microphone
            >>> spa = WorldStateRLR(mesh=...)
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

        # Parse the list of microphone aliases that we require a direct line to
        direct_path_to = self._parse_valid_microphone_aliases(ensure_direct_path)

        # If we want to express our emitters relative to a given microphone, grab this now
        desired_mic = self.get_microphone(mic) if mic is not None else None

        # Get the alias for this emitter
        alias = (
            utils.get_default_alias("src", self.emitters) if alias is None else alias
        )

        # Try and place inside the mesh: return True if placed, False if not
        placed = self._try_add_emitter(
            position, desired_mic, alias, direct_path_to, max_place_attempts
        )

        # If we can't add the emitter to the mesh
        if not placed:
            # If we were trying to add it to a random position
            if position is None:
                raise ValueError(
                    f"Could not place emitter in the mesh after {max_place_attempts} attempts. "
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

        # Update the state with the new emitter
        else:
            if self.add_to_state:
                self._update()

    def add_emitters(
        self,
        positions: Optional[Union[list, np.ndarray]] = None,
        aliases: Optional[list[str]] = None,
        mics: Optional[Union[list[str], str]] = None,
        n_emitters: Optional[int] = None,
        keep_existing: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
        raise_on_error: Optional[bool] = True,
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
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.
        """
        # Remove existing emitters if we wish to do this
        if not keep_existing:
            self.clear_emitters()

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
            desired_mic = (
                self.get_microphone(mic_alias_) if mic_alias_ is not None else None
            )

            # Get the emitter alias
            emitter_alias_ = (
                utils.get_default_alias("src", self.emitters)
                if emitter_alias_ is None
                else emitter_alias_
            )

            # Try and place the emitter inside the space
            placed = self._try_add_emitter(
                position_, desired_mic, emitter_alias_, direct_path_to
            )

            # If we can't add the emitter to the mesh
            if not placed:
                # If we were trying to add it to a random position
                if position_ is None:
                    msg = (
                        f"Could not place emitter in the mesh after {config.MAX_PLACE_ATTEMPTS} attempts. "
                        f"Consider reducing `empty_space_around` arguments."
                    )
                # If we were trying to add it to a specific position
                else:
                    msg = (
                        f"Position {position_} invalid for emitter. "
                        f"Consider reducing `empty_space_around` arguments."
                    )

                # Raise the error if required
                if raise_on_error:
                    raise ValueError(msg)

        # Update the state after placing everything
        if self.add_to_state:
            self._update()

    def get_valid_position_with_max_distance(
        self,
        ref: np.ndarray,
        r: custom_types.Numeric,
        n: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
    ) -> np.ndarray:
        """
        Generate a sphere with origin `ref` and radius `r` and sample a valid position from within its volume.

        Arguments:
            ref (np.ndarray): the reference point, treated as the origin of the sphere
            r (custom_types.Numeric): the maximum distance for the sampled point from `ref`
            n (custom_types.Numeric): the number of points to create on the sphere. Only the first valid point will be returned

        Raises:
            ValueError: if a valid point from within `n` samples cannot be found
        """
        # Input sanitization
        r = utils.sanitise_positive_number(r)
        n = utils.sanitise_positive_number(n, cast_to=int)
        ref = utils.sanitise_coordinates(ref)

        # Sample directions using normal distribution and normalize
        directions = np.random.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1)[
            :, np.newaxis
        ]  # Normalize each row

        # Sample radii with cubic root to ensure uniform volume distribution and use to scale the directions
        radii = r * np.cbrt(np.random.uniform(0, 1, size=(n,)))
        displacements = directions * radii[:, np.newaxis]

        # Add center offset
        samples = ref + displacements

        # Get a boolean mask of valid sample positions
        only_valids = self._get_valid_positions_mask(samples)
        only_valids_idxs = np.flatnonzero(only_valids)

        # If we don't have any valid samples, throw an error
        if len(only_valids_idxs) == 0:
            raise ValueError(
                f"Cannot generate a random valid point for coordinate {ref} with radius {r:.3f}. "
                f"Consider increasing the number of generated points (currently {n})"
            )
        # Otherwise, get a random valid sample and return it
        else:
            choice = np.random.choice(only_valids_idxs)
            return samples[choice, :]

    def _validate_trajectory(
        self,
        trajectory: np.ndarray,
        max_distance: custom_types.Numeric,
        step_distance: custom_types.Numeric,
        n_points: custom_types.Numeric,
        requires_direct_line_between_start_and_end: bool,
        ensure_direct_path_to_mic: Optional[list[str]] = None,
    ) -> bool:
        """
        Given a trajectory (created in `define_trajectory`), return True if valid, False otherwise

        Arguments:
            trajectory (np.ndarray): the trajectory to be validated
            requires_direct_line_between_start_and_end (bool): whether a direct line must exist between the starting and ending position
            max_distance (custom_types.Numeric): the maximum distance traversed in the trajectory, from start to end
            step_distance (custom_types.Numeric): the maximum distance traversed from one step to the next in the trajectory
            n_points (custom_types.Numeric): the expected number of points in the trajectory.
            ensure_direct_path_to_mic (list[str]): a list of microphone aliases to ensure a direct path with

        Returns:
            bool: whether the trajectory is valid
        """
        # Early return if definitely invalid
        if trajectory.shape[0] < 2:
            return False
        if trajectory.shape[0] != n_points:
            return False

        if ensure_direct_path_to_mic is None:
            ensure_direct_path_to_mic = []

        # Compute distances from start to all other points
        start = trajectory[0]
        differences = trajectory[1:] - start
        distances = np.linalg.norm(differences, axis=1)

        # If required, check that a direct path exists to all mic objects
        for d in ensure_direct_path_to_mic:
            if not all(
                self.path_exists_between_points(
                    t, self.microphones[d].coordinates_center
                )
                for t in trajectory
            ):
                return False

        # Get the furthest point from the starting position
        #  Note: for circular and linear trajectories, this should be the same as the last point.
        #  However, for random walks, this is not always the case, as we might walk very far away
        #  from the origin, and then return to it by the end of the trajectory.
        #  So, we should always consider the maximum distance from the origin, NOT the distance
        #  between the first and last point in the array
        max_idx = np.argmax(distances)
        end = trajectory[max_idx + 1]  # +1 because we sliced from [1:]

        # Check max distance constraint
        if distances[max_idx] > max_distance:
            return False

        # Optional: check for line of sight
        if (
            requires_direct_line_between_start_and_end
            and not self.path_exists_between_points(start, end)
        ):
            return False

        # Check distance between every step
        step_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        if np.any(step_deltas > step_distance + utils.SMALL):
            return False

        # Validate all positions in the trajectory WRT the rest of the mesh
        return self._validate_position(trajectory)

    def load_mesh_navigation_waypoints(
        self,
        waypoints_json: Optional[Union[Path, str]] = None,
    ) -> list[np.ndarray]:
        """
        Load the navigation waypoints for this mesh from a JSON file.

        Default filepath is <project-root>/resources/waypoints/gibson/<mesh-name>.json.
        """

        # We haven't provided a path for the JSON, so try and infer from the default location and mesh name
        if waypoints_json is None:
            mesh_fname = self.mesh.metadata["fname"]
            default_loc = utils.get_project_root() / "resources/waypoints/gibson"
            waypoints_json = (default_loc / mesh_fname).with_suffix(".json")

            # If it doesn't exist, don't worry, just skip over
            if not waypoints_json.is_file():
                logger.warning(
                    f"Cannot find waypoints for mesh {mesh_fname} inside default location ({default_loc}). "
                    f"No navigation waypoints will be loaded."
                )
                return []

        # Otherwise, check that the file we've provided exists (raises an error if not)
        else:
            waypoints_json = utils.sanitise_filepath(waypoints_json)

        # Load up the JSON and grab waypoints from it
        with open(waypoints_json, "r") as js_in:
            js_out = json.load(js_in)

        # Raise an error if not a list of dictionaries
        if not isinstance(js_out, list):
            raise ValueError(
                f"Expected waypoints JSON to be a list of dictionaries, but got type {type(js_out)}"
            )

        # Raise an error if JSON format is invalid
        elif not all("waypoints" in wp.keys() for wp in js_out):
            raise KeyError(
                "Waypoints JSON must be a list of dictionaries, each containing the key 'waypoints'."
            )

        # Load up the waypoints and validate that they are each inside the mesh
        waypoints = [
            np.array(wp["waypoints"])
            for wp in js_out
            if self._validate_position(wp["waypoints"])
        ]

        if len(waypoints) == 0:
            logger.warning("No valid navigation waypoints found!")

        return waypoints

    # @utils.timer("define trajectory")
    def define_trajectory(
        self,
        duration: custom_types.Numeric,
        starting_position: Optional[Union[np.ndarray, list]] = None,
        velocity: Optional[custom_types.Numeric] = config.DEFAULT_EVENT_VELOCITY,
        resolution: Optional[custom_types.Numeric] = config.DEFAULT_EVENT_RESOLUTION,
        shape: Optional[str] = None,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,
    ) -> np.ndarray:
        """
        Defines a trajectory for a moving sound event with specified spatial bounds and event duration.

        This method calculates a series of XYZ coordinates that outline the path of a sound event, based on the
        specified trajectory shape, the confines of the mesh, and the duration of the event. It generates a starting
        point and an end point that comply with these conditions, and then interpolates between these points according
        to the trajectory's shape.

        Optionally, a custom starting position can also be provided, and a valid ending position will be randomly
        sampled. To provide a custom ENDING position, one possibility is to provide this as the starting position, then
        invert the trajectory array returned by this function. To use both a custom start AND end position, call the
        trajectory functions in `utils.py` directly.

        Arguments:
            duration (Numeric): the length of time it should take to traverse from starting to ending position
            starting_position (np.ndarray): the starting position for the trajectory. If not provided, a random valid
                position within the mesh will be selected.
            velocity (Numeric): the speed limit for the trajectory, in meters per second
            resolution (Numeric): the number of emitters created per second
            shape (str): the shape of the trajectory; "linear", "semicircular", "random", "sawtooth", "sine"
            max_place_attempts (Numeric): the number of times to try and create the trajectory.
            ensure_direct_path: Whether to ensure a direct line exists between the emitter and given microphone(s).
                If True, will ensure a direct line exists between the emitter and ALL `microphone` objects. If a list of
                strings, these should correspond to microphone aliases inside `microphones`; a direct line will be
                ensured with all of these microphones. If False, no direct line is required for a emitter.

        Raises:
            ValueError: if a trajectory cannot be defined after `max_place_attempts`

        Returns:
            np.ndarray: the sanitised trajectory, with shape (n_points, 3)
        """
        # Compute the number of samples based on duration and resolution
        n_points = (
            utils.sanitise_positive_number(duration * resolution, cast_to=round) + 1
        )
        # Clamp `n_points` to 2, so we will always be able to create a moving trajectory
        if n_points < 2:
            n_points = 2
            logger.warning(
                f"Number of points in trajectory ({n_points}) is smaller than 2, so it is being clamped to "
                f"2 internally. If this is happening frequently, consider increasing `resolution` "
                f"(currently {resolution:.3f})."
            )

        # Sample a random shape if not given
        if shape is None:
            shape = str(np.random.choice(config.MOVING_EVENT_SHAPES))

        # Sanitise the maximum distance that we'll travel in the trajectory
        max_distance = utils.sanitise_positive_number(velocity * duration)

        # Compute the distance that we can travel in a single step
        step_limit = velocity / resolution

        # We no longer raise an error if max_distance < self.empty_space_around_emitter
        #  This is because (I think) this should parameter should only relate to emitters
        #  associated with *separate* events. We do not care if two emitters related to the
        #  *same* event are closer than this distance to each other.

        # If we've provided a starting position, sanitise and validate it before entering the loop
        if starting_position is not None:
            starting_position = utils.sanitise_coordinates(starting_position)
            if not self._validate_position(starting_position):
                raise ValueError(f"Invalid starting position ({starting_position})")

        # Parse the list of microphone aliases that we require a direct line to
        direct_path_to = self._parse_valid_microphone_aliases(ensure_direct_path)

        # Try and create the trajectory a specified number of times
        for _ in tqdm(range(max_place_attempts), desc="Placing trajectory..."):

            # If we've not provided a starting position, randomly sample one
            if starting_position is None:
                start_attempt = self.get_valid_position()
            # Otherwise, just use the starting position we've provided
            else:
                start_attempt = starting_position

            # If we're doing a random walk, there's no need to sample an ending position directly
            #  Instead, the ending position will be defined by the last point of the walk
            #  So we can just set the `end_attempt` variable to None so that the linter is happy
            if shape == "random":
                end_attempt = None
            # Try and sample a random valid position from the starting point
            else:
                try:
                    end_attempt = self.get_valid_position_with_max_distance(
                        start_attempt, max_distance, max_place_attempts
                    )
                except ValueError:
                    # Silently skip over errors in this case, so we retry with another starting position
                    if starting_position is None:
                        continue
                    # Otherwise, we need to raise the error as this starting position is invalid
                    else:
                        raise

            # Compute the trajectory with the utility function
            if shape == "linear":
                trajectory = utils.generate_linear_trajectory(
                    start_attempt, end_attempt, n_points
                )
            elif shape == "semicircular":
                trajectory = utils.generate_semicircular_trajectory(
                    start_attempt, end_attempt, n_points
                )
            elif shape == "sine":
                trajectory = utils.generate_sinusoidal_trajectory(
                    start_attempt, end_attempt, n_points
                )
            elif shape == "sawtooth":
                trajectory = utils.generate_sawtooth_trajectory(
                    start_attempt, end_attempt, n_points
                )
            elif shape == "random":
                # Unlike all other trajectories, a random walk doesn't need a predefined ending
                #  Instead, we just need to know the starting point, the number of steps,
                #  and the maximum distance any single step should take
                trajectory = utils.generate_random_trajectory(
                    start_attempt, step_limit, n_points
                )
            # We don't know what the trajectory is
            else:
                raise ValueError(
                    f"`shape` must be one of {', '.join(VALID_MOVING_EVENT_TRAJECTORIES)} but got '{shape}'"
                )

            # Validate the trajectory and return only if it is acceptable
            if self._validate_trajectory(
                trajectory,
                max_distance,
                step_limit,
                n_points=n_points,
                requires_direct_line_between_start_and_end=(
                    True if shape == "linear" else False
                ),
                ensure_direct_path_to_mic=direct_path_to,
            ):
                return trajectory

        # If we reach here, we couldn't create the trajectory
        raise ValueError(
            f"Could not define a valid movement trajectory after {max_place_attempts} attempt(s). Consider:\n"
            f"- Reducing `empty_space_around parameters`\n"
            f"- Decreasing `temporal_resolution` (currently {resolution})\n"
            f"- Increasing `max_place_attempts` (currently {max_place_attempts})\n"
            f"- Decreasing `max_distance` (currently {max_distance:.3f})"
        )

    def _add_emitters_without_validating(
        self,
        emitters: Union[list, np.ndarray],
        alias: Optional[str],
    ) -> None:
        """
        Adds emitters from a list **without checking** that they are valid.

        These emitters are assumed to be *pre-validated* (i.e., from a call to `_validate_position`), and thus no
        additional checks are performed on them here to ensure (for instance) that they are located in the mesh,
        that they are an acceptable distance away from each other, etc.

        This function is useful when adding emitters for every step in a trajectory created using `define_trajectory`:
        these individual steps may be very close to each other, and would thus be rejected when calling
        `_try_add_emitter`.
        """
        alias = (
            utils.get_default_alias("src", self.emitters) if alias is None else alias
        )

        for coord in emitters:
            emitter = Emitter(
                alias=alias, coordinates_absolute=utils.sanitise_coordinates(coord)
            )
            if alias in self.emitters:
                self.emitters[alias].append(emitter)
            else:
                self.emitters[alias] = [emitter]

        # Update the state after placing everything
        if self.add_to_state:
            self._update()

    def _simulation_sanity_check(self) -> None:
        """
        Check conditions required for simulation are met
        """
        assert (
            self.num_emitters > 0
        ), "Must have added valid emitters to the mesh before calling `.simulate`!"
        assert (
            len(self.microphones) > 0
        ), "Must have added valid microphones to the mesh before calling `.simulate`!"
        assert all(
            type(m) in MICARRAY_LIST or issubclass(type(m), MicArray)
            for m in self.microphones.values()
        ), "Non-microphone objects in microphone attribute"
        assert (
            self.ctx.get_listener_count() > 0
        ), "Must have listeners added to the ray tracing engine"
        assert (
            self.ctx.get_source_count() > 0
        ), "Must have emitters added to the ray tracing engine"
        assert (
            self.ctx.get_object_count() == 1
        ), "Must have only one mesh added to the ray tracing engine"
        # Check we have the expected number of sources and listeners
        assert (
            sum(len(em) for em in self.emitters.values()) == self.ctx.get_source_count()
        )
        assert (
            sum(m.n_listeners for m in self.microphones.values())
            == self.ctx.get_listener_count()
        )

    def simulate(self) -> None:
        """
        Simulates audio propagation in the state with the current listener and sound emitter positions.

        Note that returned IRs are NOT NORMALIZED: this occurs inside `synthesize.render_event_audio`.
        """
        # Update the ray-tracing engine with our current emitters, microphones, etc.
        self._update()
        # Sanity check that we actually have emitters and microphones in the state
        self._simulation_sanity_check()
        # Clear out any existing IRs
        self._irs = None
        # Run the simulation
        logger.info(
            f"Starting simulation with {self.num_emitters} emitters, {len(self.microphones)} microphones"
        )
        self.ctx.simulate()
        efficiency = self.ctx.get_indirect_ray_efficiency()
        # Log the ray efficiency: outdoor would have a very low value, e.g. < 0.05.
        #  A closed indoor room would have >0.95, and a room with some holes might be in the 0.1-0.8 range.
        #  If the ray efficiency is low for an indoor environment, it indicates a lot of ray leak from holes.
        logger.info(
            f"Finished simulation! Overall indirect ray efficiency: {efficiency:.3f}"
        )
        if efficiency < config.WARN_WHEN_RAY_EFFICIENCY_BELOW:
            logger.warning(
                f"Ray efficiency is below {config.WARN_WHEN_RAY_EFFICIENCY_BELOW :.0%}. It is possible that the mesh "
                f"may have holes in it. Consider decreasing `repair_threshold` when initialising the "
                f"`WorldState` object, or running `trimesh.repair.fill_holes` on your mesh."
            )
        # Format irs into a dictionary of {mic000: (N_capsules, N_emitters, N_samples), mic001: (...)}
        #  with one key-value pair per microphone. We have to do this because we cannot have ragged arrays
        #  The individual arrays can then be accessed by calling `self.irs.values()`
        self._irs = self.get_irs()

    def get_irs(self) -> OrderedDict[str, np.ndarray]:
        """
        Get the IRs from the ray-tracing context

        By default, `Context.get_audio` expects all listeners to have the same channel layout type. If, however, e.g.
        some listeners have Mono and others have Ambisonics, `Context.get_audio` will fail due to `numpy` expecting all
        arrays to have equal dims.

        Instead, we need to return a dictionary of numpy arrays.
        The output will have shape: {mic1: (N_capsules, N_emitters, N_samples), ...}
        """
        listener_counter = 0
        maxlen_samples = 0

        # Do a first pass to get the maximum length of a single IR
        for mic in self.microphones.values():
            for i in range(listener_counter, listener_counter + mic.n_listeners):
                for j in range(self.ctx.get_source_count()):
                    for k in range(self.ctx.get_ir_channel_count(i, j)):
                        ir_ijk = self.ctx.get_listener_source_channel_audio(i, j, k)
                        maxlen_samples = max(maxlen_samples, len(ir_ijk))
            listener_counter += mic.n_listeners

        # Now that we know the maximum length of an IR, we can do another pass with padding
        listener_counter = 0
        for mic in self.microphones.values():
            # By initialising an empty array, we prevent the need to explicitly call `np.pad`
            zero_arr = np.zeros(
                (mic.n_capsules, self.ctx.get_source_count(), maxlen_samples)
            )

            # Separate logic for MIC/FOA data
            if mic.channel_layout_type == "mic":
                # We need to track both the cumulative capsule number and the capsule number WRT just this mic
                for i_ctx, i_mic in zip(
                    range(listener_counter, listener_counter + mic.n_capsules),
                    range(mic.n_capsules),
                ):
                    if i_ctx >= self.ctx.get_listener_count():
                        break
                    for j in range(self.ctx.get_source_count()):
                        for k in range(self.ctx.get_ir_channel_count(i_ctx, j)):
                            ir_ijk = self.ctx.get_listener_source_channel_audio(
                                i_ctx, j, k
                            )
                            zero_arr[i_mic, j, : len(ir_ijk)] = ir_ijk
                listener_counter += mic.n_listeners

            elif (
                mic.channel_layout_type == "foa"
                or mic.channel_layout_type == "binaural"
            ):
                # Iterate over emitters
                for j in range(self.ctx.get_source_count()):
                    # Iterate over channels (FOA=4, Binaural=2)
                    for k in range(mic.n_capsules):
                        ir_ijk = self.ctx.get_listener_source_channel_audio(
                            listener_counter, j, k
                        )
                        zero_arr[k, j, : len(ir_ijk)] = ir_ijk
                # Only one listener for FOA/Binaural
                listener_counter += 1

            else:
                raise NotImplementedError(
                    "Simulation currently only supported with 'foa', 'binaural', or 'mic' channel layouts"
                )

            # Set the IRs as a property of the MicArray object
            mic.irs = zero_arr

        # Returns a dictionary with shape {mic1: (N_capsules, N_emitters, N_samples), ...}
        return OrderedDict({m_alias: m.irs for m_alias, m in self.microphones.items()})

    def create_scene(
        self,
        mic_radius: Optional[custom_types.Numeric] = 0.2,
        emitter_radius: Optional[custom_types.Numeric] = 0.1,
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

    def create_plot(self) -> plt.Figure:
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

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        # Fix bug where we haven't created the context yet
        if self.ctx is None:
            self._setup_audio_context()
            self._update()

        return dict(
            backend=self.name,
            sample_rate=self.sample_rate,
            emitters={
                s_alias: [
                    utils.coerce_nested_inputs(s_.coordinates_absolute) for s_ in s
                ]
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
            material=self.material,
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate a `WorldStateRLR` from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `WorldStateRLR`.

        Returns:
            WorldStateRLR instance.
        """

        # Validate the input
        for k in ["emitters", "microphones", "mesh", "rlr_config", "sample_rate"]:
            if k not in input_dict:
                raise KeyError(f"Missing key: '{k}'")

        # Instantiate the state
        state = cls(
            mesh=input_dict["mesh"]["fpath"],
            sample_rate=input_dict["sample_rate"],
            empty_space_around_mic=input_dict["empty_space_around_mic"],
            empty_space_around_emitter=input_dict["empty_space_around_emitter"],
            empty_space_around_surface=input_dict["empty_space_around_surface"],
            empty_space_around_capsule=input_dict["empty_space_around_capsule"],
            repair_threshold=input_dict["repair_threshold"],
            rlr_kwargs=input_dict["rlr_config"],
            material=input_dict.get("material", None),
        )

        # Instantiate the microphones and emitters from their dictionaries
        state.microphones = OrderedDict(
            {a: MicArray.from_dict(v) for a, v in input_dict["microphones"].items()}
        )
        state.emitters = OrderedDict(
            {
                a: [Emitter(alias=a, coordinates_absolute=v_) for v_ in v]
                for a, v in input_dict["emitters"].items()
            }
        )

        # Update the state so we add everything in to the ray-tracing engine
        state._update()

        return state

    def __str__(self) -> str:
        """
        Returns a string representation of the WorldState
        """
        return (
            f"'{self.__class__.__name__}' with mesh '{self.mesh.metadata['fpath']}' and "
            f"{len(self)} objects ({len(self.microphones)} microphones, {self.num_emitters} emitters)"
        )


class WorldStateSOFA(WorldState):
    """
    A WorldState where audio propagation is simulated using pre-rendered RIRs saved in a .SOFA file.

    Functionally equivalent to `SpatialScaper` and compatible with .SOFA files generated using it.
    """

    name = "SOFA"

    # When the distance between an input and matched point exceeds this value (in metres),
    #  a warning will be raised
    WARN_WHEN_DISTANCE_EXCEEDS = 0.1

    def __init__(
        self,
        sofa: Union[str, Path],
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        mic_alias: Optional[str] = None,
    ):
        super().__init__()

        # Validates the path to a sofa file
        self.sofa_path = utils.sanitise_filepath(sofa)

        # Sanitise the sample rate
        self.sample_rate = utils.sanitise_positive_number(sample_rate, cast_to=int)

        # Add a dummy microphone in to the worldstate
        # TODO: we assume only one microphone
        self.mic_alias = (
            utils.get_default_alias("mic", self.microphones)
            if mic_alias is None
            else mic_alias
        )
        self._add_dummy_microphone()

    def clear_microphones(self) -> None:
        raise NotImplementedError(
            "It is not possible to clear microphones from a 'WorldStateSOFA' object. "
            "This is because the microphones are set according to the SOFA file itself. "
            "Consider using 'WorldStateRLR' or 'WorldStateShoebox' to explicitly control the positions of microphones. "
        )

    def clear_microphone(self, alias: str) -> None:
        raise NotImplementedError(
            "It is not possible to clear a microphone from a 'WorldStateSOFA' object. "
            "This is because the microphone is set according to the SOFA file itself. "
            "Consider using 'WorldStateRLR' or 'WorldStateShoebox' to explicitly control the positions of a microphone."
        )

    def _infer_channel_layout_name(self, listener_short_name: str) -> str:
        """
        Try and infer the name of a channel layout ('foa', 'mic') from either the name of the listener given in
        the SOFA file, or the filepath.

        Returns:
            str: the name of the channel layout, or 'unknown' if can't be inferred.
        """
        # E.g., "foa", "mic", "binaural"...
        for candidate_channel_layout in CHANNEL_LAYOUT_TYPES:
            if listener_short_name == candidate_channel_layout:
                return candidate_channel_layout
            elif candidate_channel_layout in str(self.sofa_path):
                return candidate_channel_layout
        return "unknown"

    def _add_dummy_microphone(self) -> None:
        """
        Add a dummy microphone in to the WorldState at [0.0, 0.0, 0.0]
        """
        # Grab attributes from the SOFA file
        with self.sofa() as sofa:
            attrs = sofa.getGlobalAttributesAsDict()
            caps_positions = sofa.getReceiverPositionValues().data

        # Try and grab the microphone name from the listener short name and use to infer channel layout
        mic_name = attrs.get("ListenerShortName", "unknown").lower()
        clt = self._infer_channel_layout_name(mic_name)

        # Reshape the capsule positions and define the name
        # TODO: check reshaping works ok
        caps_positions = caps_positions.reshape(caps_positions.shape[:2])
        capsule_names = [str(i) for i in range(1, caps_positions.shape[0] + 1)]

        # Dynamically define a micarray class with given parameters
        marray = dynamically_define_micarray(
            name=mic_name,
            channel_layout_type=clt,
            coordinates_cartesian=caps_positions,
            capsule_names=capsule_names,
        )
        marray_init = marray()
        marray_init.set_absolute_coordinates([0.0, 0.0, 0.0])
        self.microphones[self.mic_alias] = marray_init

    @contextmanager
    def sofa(self) -> SOFAFile:
        """
        Context manager for loading and safely closing a .SOFA file.

        Usage:
            >>> with self.sofa() as sofa:
            >>>     # do something
            >>> # sofa file is now closed
        """
        loaded = SOFAFile(self.sofa_path, "r")
        if not loaded.isValid():
            raise ValueError(f"SOFA file at {self.sofa_path} is invalid!")
        try:
            yield loaded
        finally:
            loaded.close()

    def get_source_positions(self) -> np.ndarray:
        """
        Retrieves the XYZ coordinates of impulse response positions in the room.

        Returns:
            np.ndarray: An array of XYZ coordinates for the impulse response positions.
        """
        with self.sofa() as sofa:
            return sofa.getVariableValue("SourcePosition").data

    def get_listener_positions(self) -> np.ndarray:
        """
        Retrieves the XYZ coordinates of listeners in the room.

        Returns:
            np.ndarray: An array of XYZ coordinates for the listener positions.
        """
        with self.sofa() as sofa:
            return sofa.getVariableValue("ListenerPosition").data

    def get_room_min_max(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines the minimum and maximum XYZ coordinates for the current room setup.

        Returns:
            tuple: A tuple containing the minimum and maximum XYZ coordinates for the room.
        """
        all_xyz = np.vstack(
            [self.get_source_positions(), self.get_listener_positions()]
        )
        return all_xyz.min(axis=0), all_xyz.max(axis=0)

    def get_random_valid_position_idx(self) -> np.ndarray:
        """
        Get the index of a random valid position to place an object inside the state.

        Returns:
             np.ndarray: the random position to place an object inside the mesh
        """
        # First, get all source positions
        all_positions = self.get_source_positions()

        # Then, choose the index of a random position
        random_idx = np.random.randint(0, all_positions.shape[0])
        return np.array([random_idx])

    def get_nearest_source_idx(self, candidate_position: np.ndarray) -> np.ndarray:
        """
        Maps a set of trajectory points to their nearest neighbors in a given set of coordinates using a k-d tree.

        This function builds a k-d tree from a set of 3D coordinates and finds the nearest neighbor in these coordinates
        for each point in `candidate_position`

        Arguments:
            candidate_position (np.ndarray): A 2D array of points for which nearest neighbors are to be found.

        Returns:
            np.ndarray: A 2D array of nearest neighbour points, with shape (candidate_position.shape[0], 3)
        """
        # Make input 2D if required
        candidate_position = np.array(candidate_position)
        if candidate_position.ndim == 1:
            candidate_position = candidate_position[None, :]

        # Build a KDTree from the source positions
        source_positions = self.get_source_positions()
        tree = KDTree(source_positions)

        matches = []

        # Iterate over all points in the array
        for point in candidate_position:

            # Query the tree for candidate matches to this point
            distance, index = tree.query(point, k=1)

            # Coerce out of a list dtype
            if not isinstance(distance, custom_types.Numeric):
                distance = distance[0]
            if not isinstance(index, custom_types.Numeric):
                index = index[0]

            matched_point = source_positions[index]

            # Raise a warning if the distance is further away than expected
            if distance >= self.WARN_WHEN_DISTANCE_EXCEEDS:
                logger.error(
                    f"Could not find a match for point {point} within {self.WARN_WHEN_DISTANCE_EXCEEDS} metres. "
                    f"Using nearest point ({matched_point}), which is {round(distance, 2)}m away."
                )

            matches.append(index)

        return np.array(matches)

    def _try_add_emitter(
        self,
        position: Optional[Union[list, np.ndarray]],
        alias: str,
    ) -> bool:
        """
        Attempt to add a emitter at the given position with the specified alias.
        Returns True if placement is successful, otherwise False.
        """
        # First, get all source positions
        source_positions = self.get_source_positions()

        # If no position provided, get the index of a random valid position
        if position is None:
            position_idx = self.get_random_valid_position_idx()
        # Otherwise, get nearest position to the provided value
        else:
            position_idx = self.get_nearest_source_idx(position)

        # Iterate over all indices provided
        for idx in position_idx:

            # Unpack the index
            validated_position = source_positions[idx, :]

            # Log if we're using the nearest neighbour to a user-defined position
            if position is not None:
                logger.info(f"Using nearest neighbour position ({validated_position})")

            # Successfully placed: add to the emitter dictionary and return True
            emitter = Emitter(
                alias=alias,
                coordinates_absolute=utils.sanitise_coordinates(validated_position),
                sofa_idx=idx,
            )
            # Add the emitter to the list created for this alias, or create the list if it doesn't exist
            if alias in self.emitters:
                self.emitters[alias].append(emitter)
            else:
                self.emitters[alias] = [emitter]

        # TODO: under what conditions might we want to return False?
        return True

    def _add_emitters_without_validating(
        self,
        emitters: Union[list, np.ndarray],
        alias: Optional[str],
    ) -> None:
        """
        Adds emitters from a list **without checking** that they are valid.

        These emitters are assumed to be *pre-validated* (i.e., from a call to `_validate_position`), and thus no
        additional checks are performed on them here to ensure (for instance) that they are located in the mesh,
        that they are an acceptable distance away from each other, etc.

        This function is useful when adding emitters for every step in a trajectory created using `define_trajectory`:
        these individual steps may be very close to each other, and would thus be rejected when calling
        `_try_add_emitter`.
        """
        alias = (
            utils.get_default_alias("src", self.emitters) if alias is None else alias
        )

        for coord in emitters:
            coord = utils.sanitise_coordinates(coord)

            # Get sofa idx
            sofa_idx = self.get_nearest_source_idx(coord)[0]

            # Create emitter and add to state (appending to list if needed)
            emitter = Emitter(
                alias=alias,
                coordinates_absolute=utils.sanitise_coordinates(coord),
                sofa_idx=sofa_idx,
            )
            if alias in self.emitters:
                self.emitters[alias].append(emitter)
            else:
                self.emitters[alias] = [emitter]

        # Update the state after placing everything
        self._update()

    def add_emitter(
        self,
        position: Optional[Union[list, np.ndarray]] = None,
        alias: Optional[str] = None,
        mic: Optional[str] = None,  # noqa: F841
        keep_existing: Optional[bool] = False,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,  # noqa: F841
        max_place_attempts: Optional = config.MAX_PLACE_ATTEMPTS,  # noqa: F841
    ) -> None:
        """
        Add an emitter to the state.
        """
        # Remove existing emitters if we wish to do this
        if not keep_existing:
            self.clear_emitters()

        # Get the alias for this emitter
        alias = (
            utils.get_default_alias("src", self.emitters) if alias is None else alias
        )

        # Try and add the emitter with given position + alias
        placed = self._try_add_emitter(position, alias)

        # If we can't add the microphone to the mesh
        if not placed:
            # If we were trying to add it to a random position
            if position is None:
                raise ValueError("Could not find a valid position for emitter.")
            # If we were trying to add it to a specific position
            else:
                raise ValueError(f"Position {position} invalid.")

        # Update the state with the new emitter
        self._update()

    def get_valid_position_with_max_distance(
        self, ref: np.ndarray, max_distance: float
    ) -> np.ndarray:
        """
        Grab a valid position to reference that is less than `max_distance` away
        """
        # Grab all source positions
        source_positions = self.get_source_positions()

        # Compute distance WRT reference position
        distances = np.linalg.norm(source_positions - ref, axis=1)

        # Only keep those that are below the upper limit
        mask = (distances != 0) & (distances <= max_distance)
        valid_source_positions = source_positions[mask, :]

        # Randomly sample a single valid source position
        chosen_end_position_idx = np.random.randint(valid_source_positions.shape[0])
        return valid_source_positions[chosen_end_position_idx, :]

    @staticmethod
    def _validate_trajectory(
        trajectory: np.ndarray,
        max_distance: custom_types.Numeric,
        step_distance: custom_types.Numeric,
        n_points: custom_types.Numeric,
    ) -> bool:
        """
        Given a trajectory (created in `define_trajectory`), return True if valid, False otherwise

        Returns:
            bool: whether the trajectory is valid
        """
        # Early return if definitely invalid
        if trajectory.shape[0] < 2:
            return False
        if trajectory.shape[0] != n_points:
            return False

        # Compute distances from start to all other points
        start = trajectory[0]
        differences = trajectory[1:] - start
        distances = np.linalg.norm(differences, axis=1)

        # Get the furthest point from the starting position
        #  Note: for circular and linear trajectories, this should be the same as the last point.
        max_idx = np.argmax(distances)

        # Check max distance constraint
        if distances[max_idx] > max_distance:
            return False

        # Check distance between every step
        step_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        if np.any(step_deltas > step_distance + utils.SMALL):
            return False

        return True

    def define_trajectory(
        self,
        duration: custom_types.Numeric,
        starting_position: Optional[Union[np.ndarray, list]] = None,
        velocity: Optional[custom_types.Numeric] = config.DEFAULT_EVENT_VELOCITY,
        resolution: Optional[custom_types.Numeric] = config.DEFAULT_EVENT_RESOLUTION,
        shape: Optional[str] = None,
        max_place_attempts: Optional[custom_types.Numeric] = config.MAX_PLACE_ATTEMPTS,
        ensure_direct_path: Optional[Union[bool, list, str]] = False,  # noqa: F841
    ) -> np.ndarray:
        """
        Defines a trajectory for a moving sound event with specified spatial bounds and event duration.

        Returns:
            np.ndarray: the sanitised trajectory, with shape (n_points, 3)
        """
        # Compute the number of samples based on duration and resolution
        n_points = (
            utils.sanitise_positive_number(duration * resolution, cast_to=round) + 1
        )
        # Clamp `n_points` to 2, so we will always be able to create a moving trajectory
        if n_points < 2:
            n_points = 2
            logger.warning(
                f"Number of points in trajectory ({n_points}) is smaller than 2, so it is being clamped to "
                f"2 internally. If this is happening frequently, consider increasing `resolution` "
                f"(currently {resolution:.3f})."
            )

        # Sample a random shape if not given
        if shape is None:
            shape = str(np.random.choice(["linear", "semicircular"]))

        # Sanitise the maximum distance that we'll travel in the trajectory
        max_distance = utils.sanitise_positive_number(velocity * duration)

        # Compute the distance that we can travel in a single step
        step_limit = velocity / resolution

        # Grab all permissible IR positions
        source_positions = self.get_source_positions()

        # If we've provided a starting position, grab the nearest neighbour to it
        starting_position_idx = None
        if starting_position is not None:
            starting_position_idx = self.get_nearest_source_idx(starting_position)

        # Try and create the trajectory a specified number of times
        for _ in tqdm(range(max_place_attempts), desc="Placing trajectory..."):

            # If we've not provided a starting position, randomly sample one FOR THIS ATTEMPT
            if starting_position is None:
                starting_position_idx = self.get_random_valid_position_idx()

            # Get the starting position we'll use for this attempt
            start_attempt = source_positions[starting_position_idx, :]

            # Try and grab a valid ending position for this attempt
            try:
                end_attempt = self.get_valid_position_with_max_distance(
                    start_attempt, max_distance
                )
            except ValueError:
                # Silently skip over errors in this case, so we retry with another starting position
                if starting_position is None:
                    continue
                # Otherwise, we need to raise the error as this starting position is invalid
                else:
                    raise

            # Compute the trajectory with the utility function
            if shape == "linear":
                trajectory = utils.generate_linear_trajectory(
                    start_attempt, end_attempt, n_points
                )
            elif shape == "semicircular":
                trajectory = utils.generate_semicircular_trajectory(
                    start_attempt, end_attempt, n_points
                )
            else:
                raise ValueError(
                    "Only 'linear' and 'semicircular' shapes are supported"
                )

            # For every point in the trajectory, compute the nearest neighbours
            nearest_idxs = self.get_nearest_source_idx(trajectory)
            trajectory_nearest = source_positions[nearest_idxs, :]

            # Validate the trajectory
            if self._validate_trajectory(
                trajectory_nearest,
                max_distance=max_distance,
                step_distance=step_limit,
                n_points=n_points,
            ):
                return trajectory_nearest

        # If we reach here, we couldn't create the trajectory
        raise ValueError(
            f"Could not define a valid movement trajectory after {max_place_attempts} attempt(s). Consider:\n"
            f"- Decreasing `temporal_resolution` (currently {resolution})\n"
            f"- Increasing `max_place_attempts` (currently {max_place_attempts})\n"
            f"- Decreasing `max_distance` (currently {max_distance:.3f})"
        )

    def _update(self) -> None:
        """
        Updates the state, setting emitter positions correctly
        """
        # Only update when we've added emitters
        if self.num_emitters > 0:
            # Grab positions and aliases for mic
            listener_positions = self.get_listener_positions()

            # Iterate through all emitters in the state
            for emitter_alias, emitter_list in self.emitters.items():
                for emitter in emitter_list:

                    # Compute the polar and cartesian position WRT the listener
                    listener_at_idx = listener_positions[emitter.sofa_idx, :]
                    pos = emitter.coordinates_absolute - listener_at_idx

                    # Update all dictionaries with relative positions
                    # TODO: we assume only one microphone
                    emitter.coordinates_relative_cartesian[self.mic_alias] = pos
                    emitter.coordinates_relative_polar[self.mic_alias] = (
                        utils.cartesian_to_polar(pos)
                    )

    def _simulation_sanity_check(self):
        """
        Check conditions required for simulation are met
        """
        assert (
            self.num_emitters > 0
        ), "Must have added valid emitters to the state before calling `.simulate`!"
        assert len(self.microphones) == 1, "Expected only one microphone!"
        assert not any(
            [
                em.sofa_idx is None
                for em_list in self.emitters.values()
                for em in em_list
            ]
        ), "All Emitter objects must have corresponding indices in the .SOFA file"

    def simulate(self):
        """
        Grabs all required IRs for Emitters in the WorldState, resampling if necessary
        """
        # Update the ray-tracing engine with our current emitters, microphones, etc.
        self._update()
        # Sanity check that we actually have emitters and microphones in the state
        self._simulation_sanity_check()
        # Format irs into a dictionary of {mic000: (N_capsules, N_emitters, N_samples), mic001: (...)}
        #  with one key-value pair per microphone. We have to do this because we cannot have ragged arrays
        #  The individual arrays can then be accessed by calling `self.irs.values()`
        self._irs = self.get_irs()

    def get_irs(self) -> OrderedDict[str, np.ndarray]:
        """
        Get the IRs from the WorldState

        The output will have shape: {mic_alias: (N_capsules, N_emitters, N_samples)}
        """
        # Load in all IRs and their sample rate
        with self.sofa() as sofa:
            ir_sr = int(sofa.getVariableValue("Data.SamplingRate"))
            all_irs = sofa.getDataIR().data

        # Get indices of all required IRs
        required_irs = np.array(
            [em.sofa_idx for em_list in self.emitters.values() for em in em_list]
        )

        # Create empty array with shape (n_channels, n_emitters, n_samples)
        expected_out_samples = round(all_irs.shape[2] * (self.sample_rate / ir_sr))
        final_irs = np.zeros(
            (all_irs.shape[1], len(required_irs), expected_out_samples)
        )

        # Iterate over all IRs
        for total_idx, required_ir_idx in enumerate(required_irs):
            # Grab the current IR
            required_ir = all_irs[required_ir_idx, :, :]

            # Resample the IR if required
            if ir_sr != self.sample_rate:
                required_ir = librosa.resample(
                    required_ir, orig_sr=ir_sr, target_sr=self.sample_rate
                )

            # Update the zeroed array
            final_irs[:, total_idx, :] = required_ir

        return OrderedDict({self.mic_alias: final_irs})

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        with self.sofa() as sofa:
            sofa_metadata = sofa.getGlobalAttributesAsDict()

        return dict(
            backend=self.name,
            sofa=str(self.sofa_path),
            sample_rate=self.sample_rate,
            emitters={
                s_alias: [
                    utils.coerce_nested_inputs(s_.coordinates_absolute) for s_ in s
                ]
                for s_alias, s in self.emitters.items()
            },
            emitter_sofa_idxs={
                s_alias: [s_.sofa_idx for s_ in s]
                for s_alias, s in self.emitters.items()
            },
            microphones={
                m_alias: m.to_dict() for m_alias, m in self.microphones.items()
            },
            metadata={
                "bounds": [
                    utils.coerce_nested_inputs(i) for i in self.get_room_min_max()
                ],
                **sofa_metadata,
            },
        )

    @classmethod
    def from_dict(cls, input_dict: dict):
        """
        Instantiate a `WorldStateSOFA` from a dictionary.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the `WorldStateSOFA`.

        Returns:
            WorldStateSOFA instance.
        """

        # Validate the input
        for k in [
            "emitters",
            "microphones",
            "sofa",
            "metadata",
            "sample_rate",
            "emitter_sofa_idxs",
        ]:
            if k not in input_dict:
                raise KeyError(f"Missing key: '{k}'")

        # Instantiate the state
        state = cls(
            sofa=input_dict["sofa"],
            mic_alias=str(list(input_dict["microphones"].keys())[0]),
            sample_rate=input_dict["sample_rate"],
        )

        # Instantiate the microphones and emitters from their dictionaries
        # state.microphones = OrderedDict(
        #     {a: MicArray.from_dict(v) for a, v in input_dict["microphones"].items()}
        # )
        state.emitters = OrderedDict(
            {
                a: [
                    Emitter(alias=a, coordinates_absolute=v1_, sofa_idx=v2_)
                    for (v1_, v2_) in zip(v1, v2)
                ]
                for (a, v1), v2 in zip(
                    input_dict["emitters"].items(),
                    input_dict["emitter_sofa_idxs"].values(),
                )
            }
        )

        # Update the state so we add everything in
        state._update()

        return state

    def __str__(self) -> str:
        """
        Returns a string representation of the WorldState
        """
        return (
            f"'{self.__class__.__name__}' with SOFA file '{str(self.sofa_path)}' and "
            f"{len(self)} objects ({len(self.microphones)} microphones, {self.num_emitters} emitters)"
        )


class WorldStateShoebox(WorldState):
    """
    A WorldState where audio propagation is simulated inside a parameterized (user-controllable) "shoebox" room.
    """

    name = "SHOEBOX"


WORLDSTATE_LIST = [WorldStateRLR, WorldStateSOFA, WorldStateShoebox]

# Denotes a WorldState subclass
TWorldState = TypeVar("TWorldState", bound="WorldState")


def get_worldstate_from_string(worldstate_name: str) -> Type[TWorldState]:
    """
    Given a string representation of a worldstate type (e.g., `rlr`), return the correct WorldState object
    """
    # These are the name attributes for all valid WorldStates
    acceptable_values = [ws.name for ws in WORLDSTATE_LIST]
    if worldstate_name.upper() not in acceptable_values:
        raise ValueError(
            f"Cannot find array {worldstate_name}: expected one of {', '.join(acceptable_values)}"
        )
    else:
        # Using `next` avoids having to build the whole list
        return next(ws for ws in WORLDSTATE_LIST if ws.name == worldstate_name.upper())
