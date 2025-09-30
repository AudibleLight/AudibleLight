#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset and Dataloader classes, designed to be used directly with pytorch"""

from pathlib import Path
from typing import Any, Optional, Type, Union

import librosa
import librosa.feature
import numpy as np
import torch
from loguru import logger
from scipy import stats
from torch.utils.data import Dataset

from audiblelight import config, custom_types, utils
from audiblelight.core import Scene
from audiblelight.micarrays import sanitize_microphone_input
from audiblelight.synthesize import generate_dcase2024_metadata


class DatasetALight(Dataset):
    def __init__(
        self,
        scenes: list[Scene] = None,
        mesh_paths: list[Path] = None,
        n_scenes_per_mesh: int = config.N_SCENES,
        microphone_type: Optional[str] = config.MIC_ARRAY_TYPE,
        moving_events: Union[int, custom_types.DistributionLike] = None,
        static_events: Union[int, custom_types.DistributionLike] = None,
        scene_kwargs: Optional[dict] = None,
        feature_kwargs: Optional[dict] = None,
        label_kwargs: Optional[dict] = None,
    ):
        self.__iter_count = 0

        # Sanitise N moving/static events
        self.moving_events = self._sanitise_n_events(moving_events)
        self.static_events = self._sanitise_n_events(static_events)

        # Sanitise microphone type
        self.microphone_type = sanitize_microphone_input(microphone_type)

        # Arguments used in `generate_features/labels` respectively
        self.feature_kwargs = feature_kwargs if feature_kwargs is not None else {}
        self.label_kwargs = label_kwargs if label_kwargs is not None else {}

        # Passing paths rather than scenes
        if mesh_paths is not None and scenes is None:
            mesh_paths = utils.sanitise_filepaths(mesh_paths)
            n_scenes_per_mesh = utils.sanitise_positive_number(
                n_scenes_per_mesh, cast_to=int
            )

            # Repeat mesh path up to N
            self.scenes = [fp for fp in mesh_paths for _ in range(n_scenes_per_mesh)]

            # Validate kwargs for Scene.__init__
            if scene_kwargs is None:
                scene_kwargs = dict()
            utils.validate_kwargs(Scene.__init__, **scene_kwargs)
            self.scene_kwargs = scene_kwargs

        # Passing scenes rather than paths
        elif scenes is not None:
            # Coerce individual items to list
            if not isinstance(scenes, list):
                scenes = [scenes]

            # Handle invalid inputs
            if not all(isinstance(sc, Scene) for sc in scenes):
                raise TypeError(
                    "When passing `scenes`, all objects must be of type `audiblelight.core.Scene`"
                )
            self.scenes = scenes

            # Raise a warning if kwargs will be ignored
            if scene_kwargs is not None:
                logger.error(
                    "Any `scene_kwargs` will be ignored when passing `scenes` to the dataset constructor"
                )
            self.scene_kwargs = {}

            # If any scenes already have events added to them and moving/static_events not None
            #  raise a warning that no new events will be added and set these variables to None
            events_added = any(len(sc.get_events()) for sc in self.scenes)
            provided_args = any(
                (self.moving_events is not None, self.static_events is not None)
            )
            if events_added and provided_args:
                self.moving_events = None
                self.static_events = None
                logger.error(
                    "Either `moving_events` or `static_events` will be ignored when passing "
                    "`scenes` which already have Events added to the dataset constructor"
                )

        # Not sure what we've passed
        else:
            raise ValueError("Must pass one of `scenes` or `mesh_paths`!")

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self) -> Type["DatasetALight"]:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]

    def _sanitise_n_events(
        self, n_events: Any
    ) -> Union[custom_types.Numeric, custom_types.DistributionLike]:
        """
        Sanitises provided number of events: either a positive numeric value, None, or a distribution
        """
        if n_events is None:
            return self._make_uniform_distribution(
                config.MIN_MOVING_EVENTS, config.MAX_MOVING_EVENTS
            )
        elif isinstance(n_events, custom_types.Numeric):
            return utils.sanitise_positive_number(n_events, int)
        else:
            return utils.sanitise_distribution(n_events)

    @staticmethod
    def _make_uniform_distribution(
        min_: custom_types.Numeric, max_: custom_types.Numeric
    ) -> custom_types.DistributionLike:
        """
        Helper function for returning a uniform distribution between two given values
        """
        return stats.uniform(min_, max_ - min_)

    def _create_scene_with_mesh(self, mesh_path: Path) -> Scene:
        """
        Creates a Scene with a given mesh and `scene_kwargs` passed to init
        """
        return Scene(
            duration=self.scene_kwargs.get("duration", config.SCENE_DURATION),
            allow_duplicate_audios=self.scene_kwargs.get(
                "allow_duplicate_audios", False
            ),
            max_overlap=self.scene_kwargs.get("max_overlap", config.MAX_OVERLAP),
            ref_db=self.scene_kwargs.get("ref_db", config.REF_DB),
            # Paths
            mesh_path=mesh_path,
            fg_path=self.scene_kwargs.get("fg_path", config.FG_PATH),
            bg_path=self.scene_kwargs.get("bg_path", config.BG_PATH),
            # Distributions
            scene_start_dist=self.scene_kwargs.get(
                "scene_start_dist",
                self._make_uniform_distribution(0.0, config.SCENE_DURATION - 1),
            ),
            event_start_dist=self.scene_kwargs.get("event_start_dist", None),
            event_duration_dist=self.scene_kwargs.get(
                "event_duration_dist",
                self._make_uniform_distribution(
                    config.MIN_EVENT_DURATION, config.MAX_EVENT_DURATION
                ),
            ),
            event_velocity_dist=self.scene_kwargs.get(
                "event_velocity_dist",
                self._make_uniform_distribution(
                    config.MIN_EVENT_VELOCITY, config.MAX_EVENT_VELOCITY
                ),
            ),
            event_resolution_dist=self.scene_kwargs.get(
                "event_resolution_dist",
                self._make_uniform_distribution(
                    config.MIN_EVENT_RESOLUTION, config.MAX_EVENT_RESOLUTION
                ),
            ),
            snr_dist=self.scene_kwargs.get(
                "snr_dist",
                self._make_uniform_distribution(
                    config.MIN_EVENT_SNR, config.MAX_EVENT_SNR
                ),
            ),
            state_kwargs=self.scene_kwargs.get("state_kwargs", {}),
            # Augmentations
            event_augmentations=self.scene_kwargs.get("event_augmentations", []),
        )

    def _add_events_to_scene(self, scene: Scene) -> None:
        """
        Add Event objects to a Scene according to values passed in `DatasetALight.__init__`
        """

        def grabber(var_: Union[custom_types.DistributionLike, int]) -> int:
            # Sanitised to either a callable or integer
            if not isinstance(var_, int):
                return var_.rvs()
            else:
                return var_

        for _ in grabber(self.static_events):
            scene.add_event(event_type="static")
        for _ in grabber(self.moving_events):
            scene.add_event(event_type="moving")

    def _prepare_scene(self, scene: Scene) -> None:
        """
        Prepares a scene by adding microphones, events, and ambience as required, then synthesizing
        """
        # If no microphones added to the scene, add one now
        if len(scene.get_microphones()) == 0:
            scene.add_microphone(microphone_type=self.microphone_type)

        # If no events added to the scene, add these now
        if len(scene.get_events()) == 0:
            self._add_events_to_scene(scene)

        # If no ambiences added to the scene, add these now
        if len(scene.get_ambiences()) == 0:
            pass

        # If scene audio or metadata not present, generate this now (no saving of outputs)
        if len(scene.audio) == 0 or len(scene.metadata_dcase) == 0:
            scene.generate(audio=False, metadata_json=False, metadata_dcase=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scene_or_path = self.scenes[index]

        # Passed a path: generate the Scene with these arguments
        if isinstance(scene_or_path, Path):
            scene = self._create_scene_with_mesh(mesh_path=scene_or_path)

        # Passed a Scene
        elif isinstance(scene_or_path, Scene):
            scene = scene_or_path

        # Not sure what we've passed
        else:
            raise ValueError(
                f"Expected either a Path or a Scene but got {type(scene_or_path)}."
            )

        # Add anything that is missing to the scene (microphones, events, ambiences, etc.)
        #  This also runs any generation and synthesis if required
        self._prepare_scene(scene)

        # Generate the features and labels from the scene and return as a dictionary
        feat = self.generate_feature(scene)
        lab = self.generate_labels(scene)

        # TODO: at this point, we'd apply augmentation?

        # Return everything as a dictionary
        return dict(
            feature=torch.tensor(feat),
            lab=torch.tensor(lab),
        )

    @classmethod
    def from_precached(
        cls, audio_files: list[Path], json_files: list[Path]
    ) -> Type["DatasetALight"]:
        """
        Create a dataset from existing .WAV and .JSON files
        """

        # Must have equal number of audio and metadata files
        if len(audio_files) != len(json_files):
            raise ValueError(
                f"Number of audio and metadata files does not match: "
                f"got {len(audio_files)} audio files and"
                f" {len(json_files)} metadata files!"
            )

        # Sanitise audio and metadata files
        audio_files = utils.sanitise_filepaths(audio_files)
        metadata_files = utils.sanitise_filepaths(json_files)

        rendered_scenes = []
        all_sample_rate = None

        for audio, metadata in zip(audio_files, metadata_files):
            scene = Scene.from_json(metadata)

            # Raise an error if scene has more than one microphone
            if len(scene.state.microphones) > 1:
                raise NotImplementedError(
                    "Dataset currently only supports Scene objects with a single microphone"
                )

            # Parse sample rate from reconstructed scene and keep track
            sample_rate = int(scene.sample_rate)
            if all_sample_rate is None:
                all_sample_rate = sample_rate
            else:
                if all_sample_rate != sample_rate:
                    raise ValueError(
                        "All Scenes must have the same sample rate "
                        f"(expected {all_sample_rate}, got {sample_rate})"
                    )

            # Load up the audio with the given sample rate of scene and register to scene
            # TODO: allow using only `json` paths and then generate audio inside `__getitem__`
            y, _ = librosa.load(audio, sr=sample_rate, mono=False, dtype=np.float32)
            scene.audio[scene.state.microphones.keys()[0]] = y

            # Generate DCASE metadata for the scene and register it
            scene.metadata_dcase = generate_dcase2024_metadata(scene)
            rendered_scenes.append(scene)

        return cls(scenes=rendered_scenes)

    def generate_feature(self, scene: Scene) -> np.ndarray:
        """
        Generates features from a Scene.

        By default, this function returns a mel spectrogram for every channel.

        This function should also make use of any kwargs passed in to `DatasetALight.__init__(feature_kwargs=...)`

        Arguments:
            scene (Scene): a Scene to generate features from

        Returns:
            np.ndarray
        """
        mic = scene.state.microphones.keys()
        audio = scene.audio[mic]

        # Compute the spectrogram and return
        return librosa.feature.melspectrogram(
            y=audio,
            sr=int(scene.sample_rate),
            **self.feature_kwargs,
        )

    def generate_labels(self, scene: Scene) -> np.ndarray:
        """
        Generates labels for a Scene
        This function should also make use of any kwargs passed in to `DatasetALight.__init__(label_kwargs=...)`

        Arguments:
            scene (Scene): a Scene to generate features from

        Returns:
            np.ndarray
        """
        _ = self.label_kwargs
        mic = scene.state.microphones.keys()
        return scene.audio[mic]
