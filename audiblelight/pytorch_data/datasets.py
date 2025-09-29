#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset and Dataloader classes, designed to be used directly with pytorch"""

from pathlib import Path
from typing import Iterable, Optional, Type, Union

import librosa
import librosa.feature
import numpy as np
import torch
from torch.utils.data import Dataset

from audiblelight import config, custom_types, utils
from audiblelight.augmentation import SceneAugmentation
from audiblelight.core import Scene
from audiblelight.synthesize import generate_dcase2024_metadata


class _DatasetALight(Dataset):
    """
    A `Dataset` for loading or generating `Scene` objects during training.

    This class can be used for either generating `Scene` audio features on the fly when iterating it,
    or by pre-generating all the files prior to initializing the dataset and then loading the data
    up as JSON metadata and WAV audio files.

    `SceneAugmentation` objects can be passed to the `Dataset` object in order to apply augmentations
    to the output.

    **Important note**: the `generate_audio_feature` method can (and should) be overwritten in order to return
    custom features from this dataset.
    """

    def __init__(
        self,
        scenes: list[Scene],
        window_size: Optional[int] = config.WIN_SIZE,
        hop_size: Optional[int] = config.HOP_SIZE,
    ):
        self.__iter_count = 0
        self.window_size = utils.sanitise_positive_number(window_size, cast_to=int)
        self.hop_size = utils.sanitise_positive_number(hop_size, cast_to=int)
        self.scenes = scenes

        # Parse sample rate from scenes
        if not all(s.sample_rate for s in self.scenes == self.scenes[0].sample_rate):
            raise ValueError("Not all scenes have the same sample rate!")
        self.sample_rate = utils.sanitise_positive_number(
            self.scenes[0].sample_rate, cast_to=int
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def compute_frame_count(self, audio: np.ndarray) -> int:
        """
        Compute number of frames for an audio input

        Parameters:
            audio (np.ndarray): audio input

        Returns:
            int: number of frames in the input, given window and hop size
        """
        # Compute number of valid frames
        return (audio.shape[1] - self.window_size) // self.hop_size + 1

    def compute_frame_info(self, frame_num: int) -> tuple[int, int, int, int]:
        """
        Given the number of a frame (e.g., 1, 3, 5), compute start + end time of the frame in samples + ms
        """
        # Compute start and end in samples
        start_sample = round(frame_num * self.hop_size)
        end_sample = start_sample + self.window_size
        # Compute start and end in milliseconds, rounded to nearest integer value
        start_ms = round((start_sample / self.sample_rate) * 1000)
        end_ms = round((end_sample / self.sample_rate) * 1000)
        return start_sample, end_sample, start_ms, end_ms

    def generate_feature(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generates features from spatial audio

        By default, this function returns a mel spectrogram for every channel, parsing the parameters from **kwargs.

        Arguments:
            audio (np.ndarray): spatial audio rendered from a scene
            kwargs: additional keyword arguments used in the method

        Returns:
            np.ndarray
        """
        return librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, **kwargs)

    def generate_labels(self, labels: np.ndarray, **_) -> np.ndarray:
        """
        Generates labels from DCASE-style metadata

        By default, this function just returns an array of (se_label, x, y) for every currently active class

        Arguments:
            labels (np.ndarray): labels from DCASE-style metadata, i.e. `synthesize.generate_dcase2024_metadata()`
        """
        # Just returns the class ID, azimuth, and elevation
        return labels[:, [1, 3, 4]]

    def __iter__(self) -> Type["_DatasetALight"]:
        return self

    def __next__(self):
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]


class DatasetCached(_DatasetALight):
    """
    A `Dataset` for loading precomputed `Scene` objects during training.

    Every `Scene` object should be defined by:
        1) a `wav` file on the disk, pointing to the audio output of `Scene.generate`
        2) a `JSON` file on the disk, pointing to the output of `Scene.to_json`

    `SceneAugmentation` objects can also be passed in order to apply augmentations to the output.

    **Important note**: the `generate_feature/label` methods can (and should) be overwritten in order to return
    custom features and labels for every audio frame. By default, these functions simply return mel spectrograms and
    the class IDs and DOA vectors for each sound event active in a frame.
    """

    def __init__(
        self,
        audio_files: list[Union[Path]],
        metadata_files: list[Union[Path]],
        window_size: Optional[int] = config.WIN_SIZE,
        hop_size: Optional[int] = config.HOP_SIZE,
        scene_augmentations: Optional[
            Union[Type[SceneAugmentation], Iterable[Type[SceneAugmentation]]]
        ] = None,
    ):
        scenes, self.frame_indices = zip(*self.load_scenes(audio_files, metadata_files))
        super().__init__(scenes, window_size, hop_size)

    def load_scenes(
        self,
        audio_files: list[Union[Path]],
        metadata_files: list[Union[Path]],
    ) -> list[Scene]:
        """
        Load up all pre-cached scenes from audio and metadata files
        """
        # Must have equal number of audio and metadata files
        if len(audio_files) != len(metadata_files):
            raise ValueError(
                f"Number of audio and metadata files does not match: "
                f"got {len(audio_files)} audio files and"
                f" {len(metadata_files)} metadata files!"
            )

        # Sanitise audio and metadata files
        audio_files = utils.sanitise_filepaths(audio_files)
        metadata_files = utils.sanitise_filepaths(metadata_files)

        # Create Scene objects from metadata + audio files
        rendered_scenes = []
        for audio, metadata in zip(audio_files, metadata_files):
            scene = Scene.from_json(metadata)

            # Raise an error if scene has more than one microphone
            if len(scene.state.microphones) > 1:
                raise NotImplementedError(
                    "Dataset currently only supports Scene objects with a single microphone"
                )

            # Load up the audio with the given sample rate of scene and register to scene
            y, _ = librosa.load(
                audio, sr=int(scene.sample_rate), mono=False, dtype=np.float32
            )
            scene.audio[scene.state.microphones.keys()[0]] = y

            # Generate DCASE metadata for the scene and register it
            scene.metadata_dcase = generate_dcase2024_metadata(scene)

            # Compute the number of frames we should expect to receive from this scene
            for frame_idx in self.compute_frame_count(audio):
                rendered_scenes.append((scene, frame_idx))

        return rendered_scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Unpack the current scene and frame
        scene = self.scenes[index]
        frame_num = self.frame_indices[index]

        # Convert frame number to start/end samples/ms
        start_sample, end_sample, start_ms, end_ms = self.compute_frame_info(frame_num)

        # Only considering a single mic for now
        mic = scene.state.microphones.keys()[0]

        # Grab the audio and truncate to the current frame, then generate the feature
        audio = scene.audio[mic][start_sample:end_sample]
        feature = self.generate_feature(audio)

        # Grab the metadata and truncate to the current frame, then generate the labels
        meta = scene.metadata_dcase[mic].reset_index(drop=False).to_numpy()
        frame_idxs = meta[:, 0]
        start_idx = np.argmin(frame_idxs == start_ms)
        end_idx = np.argmax(frame_idxs == end_ms)
        labels = self.generate_labels(meta[start_idx:end_idx, :])

        return {
            "features": torch.tensor(feature),
            "labels": torch.tensor(labels),
        }


class DatasetMeshes(_DatasetALight):
    def __init__(
        self,
        mesh_paths: Iterable[Path],
        n_scenes: Optional[int] = config.N_SCENES,
        static_events: Optional[
            Union[int, custom_types.DistributionLike]
        ] = config.DEFAULT_STATIC_EVENTS,
        moving_events: Optional[
            Union[int, custom_types.DistributionLike]
        ] = config.DEFAULT_MOVING_EVENTS,
        scene_kwargs: Optional[dict] = None,
        static_event_kwargs: Optional[dict] = None,
        moving_event_kwargs: Optional[dict] = None,
        ambience_kwargs: Optional[dict] = None,
    ):
        """
        Generates Scene objects "on-the-fly" from a list of mesh paths.

        Arguments:
            mesh_paths: list of paths to mesh files
            n_scenes: number of scenes to create per mesh
            moving_events: number of moving events per scene, or a distribution to sample from (must return integers)
            static_events: number of static events per scene, or a distribution to sample from (must return integers)
            scene_kwargs: keyword arguments passed to `Scene.__init__`
            static_event_kwargs: keyword arguments passed to `Scene.add_event_static`
            moving_event_kwargs: keyword arguments passed to `Scene.add_event_moving`
            ambience_kwargs: keyword arguments passed to `Scene.add_ambience`
        """
        super().__init__()

        # Validate all kwargs
        utils.sanitise_filepaths(mesh_paths)

        # Combine mesh paths with n_scenes
        meshes = []
        for mesh_path in mesh_paths:
            for _ in range(n_scenes):
                meshes.append(mesh_path)
        self.mesh_paths = utils.sanitise_filepaths(mesh_paths)

        # Validate keyword arguments
        utils.validate_kwargs(Scene.__init__, **scene_kwargs)
