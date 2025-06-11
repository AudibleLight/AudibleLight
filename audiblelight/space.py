#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides classes and functions for representing triangular meshes, handling spatial operations, generating RIRs."""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from rlr_audio_propagation import Config, Context, ChannelLayout, ChannelLayoutType
from netCDF4 import Dataset
from loguru import logger

from audiblelight import utils

FACE_FILL_COLOR = [255, 0, 0, 255]

MIN_AVG_RAY_LENGTH = 3.0
MAX_PLACE_ATTEMPTS = 100    # Max number of times we'll attempt to place a source or microphone before giving up

MIN_DISTANCE_FROM_SOURCE = 0.2  # Minimum distance one sound source can be from another
MIN_DISTANCE_FROM_MIC = 0.1    # Minimum distance one sound source can be from the mic
MIN_DISTANCE_FROM_SURFACE = 0.2    # Minimum distance from the nearest mesh surface


def load_mesh(mesh: Union[str, Path, trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    Loads a mesh from disk or directly from a `trimesh.Trimesh` object
    """
    # Passed in a filepath
    if isinstance(mesh, (str, Path)):
        if not os.path.exists(mesh):
            raise FileNotFoundError(f"Cannot find mesh file at {mesh}, does it exist?")
        # elif not mesh.endswith(".glb"):
        #     logger.warning(f"Mesh file {mesh} is not a .glb file!")
        # Load mesh from file
        loaded_mesh = trimesh.load_mesh(mesh, file_type=mesh.split(".")[-1])
    # Passed in a loaded mesh object
    elif isinstance(mesh, trimesh.Trimesh):
        loaded_mesh = mesh
    # Passed in something else
    else:
        raise TypeError(f"Expected mesh to be either a filepath or Trimesh object, but got {type(mesh)}")
    return loaded_mesh


def validate_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Validates watertight status of a mesh and repairs when necessary
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Get the idxs of the faces in the mesh which break the watertight status of the mesh.
    broken_faces = trimesh.repair.broken_faces(new_mesh, color=FACE_FILL_COLOR)
    # Only run the repairing function when there are actually broken faces
    if len(broken_faces) > 0:
        logger.warning(f"Found {len(broken_faces)} broken faces in mesh, repairing...")
        repair_mesh(new_mesh)
    return new_mesh


def repair_mesh(mesh: trimesh.Trimesh) -> None:
    """
    Uses trimesh functionality to repair a mesh inplace
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
        mic_position (np.array): Position of the microphone in the mesh.
        ctx (rlr_audio_propagation.Context): The context for audio propagation simulation.
        source_positions (np.array): relative positions of sound sources

    """
    def __init__(self, mesh: str | trimesh.Trimesh, mic_position: np.ndarray = None):
        """
        Initializes the Space with a mesh and optionally a specific microphone position, and sets up the audio context.

        Args:
            mesh (str|trimesh.Trimesh): The name of the mesh file (without file extension).
            mic_position (np.array, optional): Initial position of the microphone within the mesh.
                If None, a random valid position will be generated.
        """
        # Store source positions in here to access later; these should be in ABSOLUTE form
        self.source_positions: np.ndarray = np.array([])

        # Initialize mesh and filename
        if isinstance(mesh, str):
            self.mesh_fpath = mesh
        else:
            self.mesh_fpath = ""
        mesh = load_mesh(mesh)
        self.mesh = validate_mesh(mesh)

        # Setting up audio context
        # TODO: is it possible to set the sample rate here?
        cfg = Config()
        self.ctx = Context(cfg)
        self.setup_audio_context()
        # Adding the microphone position if provided by the user, and it's inside the 3D mesh.
        self.mic_position = self.setup_microphones(mic_position)
        # Setting up listener
        self.setup_listener()

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
        # Compute weights by squaring the distances to give more importance to longer rays
        weights = distances ** 2
        # Calculate weighted average of the distances using the computed weights
        weighted_average = np.sum(distances * weights) / np.sum(weights)
        # Return the weighted average ray length
        return weighted_average

    def setup_audio_context(self) -> None:
        """
        Initializes the audio context and configures the mesh for the context.
        """
        self.ctx.add_object()
        self.ctx.add_mesh_vertices(self.mesh.vertices.flatten().tolist())
        self.ctx.add_mesh_indices(self.mesh.faces.flatten().tolist(), 3, "default")
        self.ctx.finalize_object_mesh(0)

    def setup_listener(self) -> None:
        """
        Adds a listener to the audio context and sets its position to the microphone's position.
        WARNING: Hard-coded to AMBISONICS for now
        """
        # TODO: remove hardcoded ambisonics
        # TODO: this should be dependent on the number of microphones that we have, so with 2 microphones == 2 listeners
        self.ctx.add_listener(ChannelLayout(ChannelLayoutType.Ambisonics, 4))
        self.ctx.set_listener_position(0, self.mic_position.tolist())

    def setup_microphones(self, mic_pos: Union[list[np.ndarray], np.ndarray, None]) -> np.ndarray:
        """
        Add the microphone position if provided by the user, and it's inside the 3D mesh.
        """
        # When no position passed in, always used a random position
        if not mic_pos:
            return self.get_random_point_inside_mesh()
        # Coerce list types to arrays
        if isinstance(mic_pos, list):
            mic_pos = np.array(mic_pos)
        # If the array is not 1D, just use the first position
        # TODO: add support for multiple microphones
        if len(mic_pos.shape) > 1:
            mic_pos = mic_pos[:, 0]
        # If the array is inside the mesh, we just use the given position
        if self._is_point_inside_mesh(mic_pos):
            placed = mic_pos
        # Otherwise, we need to get a random position inside the array
        else:
            logger.warning(f"Provided microphone position {mic_pos} not inside mesh, using a random mic position")
            placed = self.get_random_point_inside_mesh()
        return placed

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

    def get_random_point_inside_mesh(self, min_distance_from_surface: float = MIN_DISTANCE_FROM_SURFACE) -> np.ndarray:
        """
        Generates a random point inside the mesh that adheres to a minimum distance from the mesh surface.

        Args:
            min_distance_from_surface (float): The minimum required distance from the nearest surface of the mesh.

        Returns:
            np.array: A valid point within the mesh.
        """
        while True:
            point = np.random.uniform(self.mesh.bounds[0], self.mesh.bounds[1])
            if self._is_point_inside_mesh(point):
                _, distance, _ = self.mesh.nearest.on_surface([point])
                if distance[0] >= min_distance_from_surface:
                    return point

    def _is_point_inside_mesh(self, point: np.array) -> bool:
        """
        Determines whether a given point is inside the mesh.

        Args:
            point (np.array): The point to check.

        Returns:
            bool: True if the point is inside the mesh, otherwise False.
        """
        shape = len(point.shape)
        if shape == 1:
            point = np.array([point])
        elif shape > 2:
            raise ValueError(f"Expected shape == 2, but got {shape}")
        return self.mesh.contains(point)[0]

    def _set_sources(self):
        """
        Sets the positions of sound sources in the space.
        """
        self.ctx.clear_sources()
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
            (np.linalg.norm(pos_abs - pos) >= MIN_DISTANCE_FROM_SOURCE for pos in self.source_positions),
            # source must be a minimum distance from the microphone
            np.linalg.norm(pos_abs - self.mic_position) >= MIN_DISTANCE_FROM_MIC,
            # source must be inside the mesh
            self._is_point_inside_mesh(pos_abs)
        ))

    def add_sources(self, source_positions: list[np.ndarray]) -> None:
        """
        Adds pre-defined sources to a space, skipping over when they are invalid.

        Args:
            source_positions (List[np.array]): A list of source positions, relative to the microphone
        """
        valid_source_positions = []    # clear the list of sources
        for i, pos in enumerate(source_positions):
            # Position is relative, rather than absolute
            # TODO: account for multiple microphone setups here
            pos = self.mic_position + np.asarray(pos)
            # Validate that the point is inside the mesh
            if self._validate_source_position(pos):
                valid_source_positions.append(pos)
            else :
                logger.warning(f"Source {i} located outside of mesh with position {pos}. "
                               f"Skipping source {i} and moving on to source {i + 1}")
                continue
        self.source_positions = np.asarray(source_positions)
        self._set_sources()

    def add_random_sources(
            self,
            n_sources: int,
    ) -> None:
        """
        Adds N random sound sources to the space,

        Sources must be at least MIN_DISTANCE_FROM_MIC away from the microphone and MIN_DISTANCE_FROM_SOURCE away
        from each other.

        Args:
            n_sources (int): The number of sources to place randomly in the space.
        """

        source_positions = []    # clear the list of sources
        # Iterate over the number of sources we want to try and place
        for source_idx in range(n_sources):
            # Iterate over the number of times we try and place the source
            for attempt in range(MAX_PLACE_ATTEMPTS):
                source_pos_abs = self.get_random_position()
                # If the source is in a valid position, add it to the list
                if self._validate_source_position(source_pos_abs):
                    source_positions.append(source_pos_abs)
                    break
                # If we've tried too many times to place the source, make a log
                elif attempt == MAX_PLACE_ATTEMPTS - 1:
                    logger.error(f"Could not place source {source_idx}, skipping to source {source_idx + 1}. "
                                 f"If this is happening frequently, consider reducing `n_sources`, "
                                 f"`min_distance_from_mic`, or `min_distance_from_source`.")
        # Add all the valid sources in
        self.source_positions = np.asarray(source_positions)
        self._set_sources()

    def simulate(self) -> np.ndarray:
        """
        Simulates audio propagation in the space with the current listener and source positions.

        Returns:
            np.array: The impulse response matrix from the simulation.
        """
        self.ctx.simulate()
        return self.retrieve_impulse_responses()

    @lru_cache(maxsize=None)
    def retrieve_impulse_responses(self) -> np.ndarray:
        """
        Retrieves impulse responses from the context after simulation. Cached for speed on multiple calls.

        Returns:
            np.array: A matrix containing the impulse responses for the listener-source setup.
        """
        # IRs of shape (n_listeners, n_sources, n_channels, n_samples)
        # https://github.com/beasteers/rlr-audio-propagation/blob/f510714ffe3a968555754d49edc1bfdcee8d4f5f/rlr_audio_propagation/core.py#L70
        irs = self.ctx.get_audio()
        # We return [0] here as we only added one listener here
        # TODO: will need to be adapted for multiple microphone array setups
        return irs[0]

    def save_sofa(self, outpath: str) -> None:
        """
        Dumps room impulse responses to SOFA format.

        Args:
            outpath (str): path to dump .SOFA file to
        """
        # Load in IRs and unpack shape
        rirs = self.retrieve_impulse_responses()
        M, R, N = rirs.shape
        E, I, C = 1, 1, 3
        # Sanity checking
        assert rirs.shape == (M, R, N), f"RIRs shape mismatch: expected {(M, R, N)}, got {rirs.shape}"
        assert self.source_positions.shape == (M, C), f"Source position shape mismatch: expected {(M, C)}, got {self.source_positions.shape}"
        # Skip over creating files that already exist
        if os.path.exists(outpath):
            logger.warning(f"File {outpath} already exists, skipping save!")
            return
        # Create dataset
        rootgrp = Dataset(outpath, "w", format="NETCDF4")
        # Save required attributes
        rootgrp.Conventions = "SOFA"
        # rootgrp.Version = "2.1"
        rootgrp.SOFAConventions = "SingleRoomSRIR"
        rootgrp.SOFAConventionsVersion = "1.0"
        rootgrp.APIName = "pysofaconventions"
        rootgrp.APIVersion = "0.1.5"
        rootgrp.DataType = "FIR"
        current_time = time.ctime(time.time())
        rootgrp.DateCreated = current_time
        rootgrp.DateModified = current_time
        # rootgrp.AuthorContact = "chris.ick@nyu.edu"
        # rootgrp.Organization = "Music and Audio Research Lab - NYU"
        # rootgrp.License = "Use whatever you want"
        # rootgrp.Title = db_name + " - " + self.mesh_fpath
        # rootgrp.RoomType = "shoebox"
        # rootgrp.DatabaseName = db_name
        # rootgrp.ListenerShortName = "mic"
        # rootgrp.RoomShortName = self.mesh_fpath
        # Create all dimensions
        for dim_str, dim_val in zip(["M", "N", "E", "R", "I", "C"], [M, N, E, R, I, C]):
            rootgrp.createDimension(dim_str, dim_val)
        # Create listener position variable
        listener_pos_var = rootgrp.createVariable("ListenerPosition", "f8", ("M", "C"))
        listener_pos_var.Units = "metre"
        listener_pos_var.Type = "cartesian"
        listener_pos_var[:] = self.mic_position
        # Create listener up variable
        listener_up_var = rootgrp.createVariable("ListenerUp", "f8", ("I", "C"))
        listener_up_var.Units = "metre"
        listener_up_var.Type = "cartesian"
        listener_up_var[:] = np.asarray([0, 0, 1])
        # Create listener view variable
        listener_view_var = rootgrp.createVariable("ListenerView", "f8", ("I", "C"))
        listener_view_var.Units = "metre"
        listener_view_var.Type = "cartesian"
        listener_view_var[:] = np.asarray([1, 0, 0])
        # Create emitter position variable
        emitter_position_var = rootgrp.createVariable(
            "EmitterPosition", "f8", ("E", "C", "I")
        )
        emitter_position_var.Units = "metre"
        emitter_position_var.Type = "spherical"
        emitter_position_var[:] = np.zeros((E, C, I))
        # Create source position variable
        source_position_var = rootgrp.createVariable("SourcePosition", "f8", ("M", "C"))
        source_position_var.Units = "metre"
        source_position_var.Type = "cartesian"
        source_position_var[:] = self.source_positions
        # Create source up variable
        source_up_var = rootgrp.createVariable("SourceUp", "f8", ("I", "C"))
        source_up_var.Units = "metre"
        source_up_var.Type = "cartesian"
        source_up_var[:] = np.asarray([0, 0, 1])
        # Create source view variable
        source_view_var = rootgrp.createVariable("SourceView", "f8", ("I", "C"))
        source_view_var.Units = "metre"
        source_view_var.Type = "cartesian"
        source_view_var[:] = np.asarray([1, 0, 0])
        # Create receiver position variable
        receiver_position_var = rootgrp.createVariable(
            "ReceiverPosition", "f8", ("R", "C", "I")
        )
        receiver_position_var.Units = "metre"
        receiver_position_var.Type = "cartesian"
        receiver_position_var[:] = np.zeros((R, C, I))
        # Create sampling rate variable
        sampling_rate_var = rootgrp.createVariable("Data.SamplingRate", "f8", ("I",))
        sampling_rate_var.Units = "hertz"
        sampling_rate_var[:] = utils.SAMPLE_RATE
        # Create delay variable
        delay_var = rootgrp.createVariable("Data.Delay", "f8", ("I", "R"))
        delay_var[:, :] = np.zeros((I, R))
        # Create RIR variable
        data_ir_var = rootgrp.createVariable("Data.IR", "f8", ("M", "R", "N"))
        data_ir_var.ChannelOrdering = "acn"  # standard ambi ordering
        data_ir_var.Normalization = "sn3d"
        data_ir_var[:] = rirs
        # Close file to finish
        rootgrp.close()
        logger.info(f"SOFA file saved to {outpath}")

    def create_scene(self) -> trimesh.Scene:
        """Creates a trimesh.Scene with the Space's mesh, microphone position, and sources all added"""
        # Finally, visualize microphone and sources inside the scene
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh)
        # This just adds the microphone into the scene
        add_sphere(scene, self.mic_position, color=[255, 0, 0], r=0.2)
        # This adds the sound sources, with different color + radius
        for source in self.source_positions:
            add_sphere(scene, source, [0, 255, 0], r=0.1)
        return scene    # can then run `.show()` on the returned object

    def plot_scene(self,) -> plt.Figure:
        """Creates a matplotlib.Figure object corresponding to top-down and side-views of the scene"""
        # Create a figure with two subplots side by side
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        vertices = self.mesh.vertices
        # Create a top-down view first, then a side view
        for ax_, idx, color, ylab, title in zip(ax.flatten(), [1, 2], ["red", "blue"], ["Y", "Z"], ["Top", "Side"]):
            # Scatter the vertices first
            ax_.scatter(vertices[:, 0], vertices[:, idx], c='gray', alpha=0.1, s=1)
            # Then the microphone and source positions
            ax_.scatter(self.mic_position[0], self.mic_position[idx], c='red', s=100, label='Microphone')
            ax_.scatter(
                self.source_positions[:, 0], self.source_positions[:, idx],
                c='blue', s=25, alpha=0.5, label='Sound Sources'
            )
            # These are just plot aesthetics
            ax_.set_xlabel('X')
            ax_.set_ylabel(ylab)
            ax_.set_title(f'{title} view of {os.path.basename(self.mesh_fpath)}')
            ax_.legend()
            ax_.axis('equal')
            ax_.grid(True)
        # Return the matplotlib figure object
        fig.tight_layout()
        return fig    # can be used with plt.show, fig.savefig, etc.


if __name__ == "__main__":
    import glob
    from scipy.io import wavfile

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
        room = Space(mesh=mesh_path, mic_position=None)
        room.add_random_sources(5)
        # Simulate the room impulse responses
        irs_ = room.simulate()
        # Visualize microphone and sources inside the scene
        room_scene = room.create_scene()
        room_fig = room.plot_scene()
        # Dump the plot to an image and the RIRs to a SOFA file
        basename = os.path.dirname(mesh_path)
        room_fig.savefig(os.path.join(basename, "out.png"))
        room.save_sofa(os.path.join(basename, "out.sofa"))
        # Write out the IRs to a simple stereo file
        for idx_, ir in enumerate(irs_):
            ir_stereo = utils.foa_to_simple_stereo(ir)
            ir_fpath = os.path.join(basename, f"ir_{str(idx_).zfill(3)}.wav")
            wavfile.write(ir_fpath, utils.SAMPLE_RATE, ir_stereo.T)
