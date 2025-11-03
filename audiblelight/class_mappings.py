#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides mappings converting from class labels to indices for different tasks"""

from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

from loguru import logger


class ClassMapping:
    """
    Parent class for all class mapping objects
    """

    YEAR = None
    TASK = None

    def __init__(self, mapping: Optional[dict[str, int]] = None):
        self._mapping = mapping or {}
        self.validate_mapping()

    @property
    def mapping(self) -> dict[str, int]:
        """
        Returns a mapping of "class_name": "class_index"
        """
        return self._mapping

    @property
    def mapping_inverted(self) -> dict[int, str]:
        """
        Returns an inverted mapping, from "class_index": "class_name
        """
        return {v: k for k, v in self.mapping.items()}

    def infer_label_idx_from_filepath(
        self, filepath: Union[Path, str]
    ) -> Union[tuple[int, str], tuple[None, None]]:
        """
        Given a filepath, infer the class label and index from this.

        Returns a tuple of None, None if the class label cannot be inferred.

        Arguments:
            filepath: the path to infer the class label from

        Examples:
            >>> fpath = "/AudibleLight/resources/soundevents/maleSpeech/train/Male_speech_and_man_speaking/67669.wav"
            >>> mapping = ClassMapping()
            >>> c, i = mapping.infer_label_idx_from_filepath(fpath)
            >>> print(c, i)
            tuple("maleSpeech", 1)
        """
        # Coerce paths
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        cls, idx = None, None
        # Search for the label key in the path
        for part in filepath.parts:
            if part in self.mapping.keys():

                # Update the variables, only if we haven't already done so
                if not cls and not idx:
                    cls = part
                    idx = self[cls]

                # Raise if we have multiple possible matches
                else:
                    raise ValueError(
                        f"Found multiple possible classes for filepath {str(filepath)}. "
                        f"This filepath matches both classes {cls} and {part}. "
                        f"Please adjust your filepaths as necessary so that they only contain one class."
                    )

        # Raise a warning if no class index or label found
        if idx is None or cls is None:
            logger.warning(
                f"Could not find a matching class index and label for file {str(filepath)}!"
            )

        # This will just return None, None if no matches
        return idx, cls

    def infer_missing_values(
        self, class_id: Optional[int], class_label: Optional[str]
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Infers missing class ID or label if only one is provided.

        - If only class_label is provided, class_id is inferred.
        - If only class_id is provided, class_label is inferred.
        - If both are provided or both are None, returns them as-is.
        """
        if class_id is None and class_label is not None:
            class_id = self[class_label]

        elif class_id is not None and class_label is None:
            class_label = self[class_id]

        return class_id, class_label

    def __len__(self) -> int:
        """
        Returns the number of classes for this mapping
        """
        return len(self.mapping)

    def __getitem__(self, item: Any) -> Any:
        """
        Convert a class name into class index, and vice-versa
        """
        if item in self.mapping.keys():
            return self.mapping[item]
        elif item in self.mapping_inverted.keys():
            return self.mapping_inverted[item]
        else:
            raise KeyError(f"Item {item} is not a valid key or value")

    @classmethod
    def from_dict(cls, input_dict: dict[str, int]):
        """
        Compute a class mapping from a dictionary
        """
        return cls(mapping=input_dict)

    def to_dict(self) -> dict[str, int]:
        """
        Returns the class mapping as a dictionary
        """
        return self.mapping

    def validate_mapping(self) -> None:
        """
        Validates mapping after class initialization
        """

        # Check overall type
        if not isinstance(self.mapping, dict):
            raise TypeError(f"Mapping must be a dict, but got {type(self.mapping)}.")

        # Check item types
        for k, v in self.mapping.items():
            if not isinstance(k, str):
                raise TypeError(f"Class name must be str, got {type(k).__name__}: {k}")
            if not isinstance(v, int):
                raise TypeError(f"Class index must be int, got {type(v).__name__}: {v}")

        # Check for duplicate values
        indices = list(self.mapping.values())
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate indices detected.")

        # Check for non-contiguous indices
        if sorted(indices) != list(range(min(indices), max(indices) + 1)):
            raise ValueError("Indices must be contiguous from 0..N-1.")


class DCASE2023Task3(ClassMapping):
    """
    Class mappings used for DCASE2023, task 3.

    See https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#sound-event-classes
    """

    YEAR = 2023
    TASK = 3

    @property
    def mapping(self) -> dict[str, int]:
        return {
            "femaleSpeech": 0,
            "maleSpeech": 1,
            "clapping": 2,
            "telephone": 3,
            "laughter": 4,
            "domesticSounds": 5,
            "footsteps": 6,
            "doorCupboard": 7,
            "music": 8,
            "musicInstrument": 9,
            "waterTap": 10,
            "bell": 11,
            "knock": 12,
        }


class DCASE2021Task3(ClassMapping):
    """
    Class mappings used for DCASE2021, task 3.

    See https://dcase.community/challenge2021/task-sound-event-localization-and-detection#sound-event-classes
    """

    YEAR = 2021
    TASK = 3

    @property
    def mapping(self) -> dict[str, int]:
        return {
            "alarm": 0,
            "baby": 1,
            "crash": 2,
            "dog": 3,
            "femaleScream": 4,
            "femaleSpeech": 5,
            "footsteps": 6,
            "knock": 7,
            "maleScream": 8,
            "maleSpeech": 9,
            "phone": 10,
            "piano": 11,
        }


class DCASE2020Task3(ClassMapping):
    """
    Class mappings used for DCASE2020, task 3.

    See https://dcase.community/challenge2020/task-sound-event-localization-and-detection#sound-event-classes
    """

    YEAR = 2020
    TASK = 3

    @property
    def mapping(self) -> dict[str, int]:
        return {
            "alarm": 0,
            "baby": 1,
            "crash": 2,
            "dog": 3,
            "engine": 4,
            "femaleScream": 5,
            "femaleSpeech": 6,
            "fire": 7,
            "footsteps": 8,
            "knock": 9,
            "maleScream": 10,
            "maleSpeech": 11,
            "phone": 12,
            "piano": 13,
        }


class DCASE2025Task4(ClassMapping):
    """
    Class mappings used for DCASE2025, task 4.

    See https://dcase.community/challenge2025/task-spatial-semantic-segmentation-of-sound-scenes#audio-dataset
    """

    YEAR = 2025
    TASK = 4

    @property
    def mapping(self) -> dict[int, str]:
        return {
            "AlarmClock": 0,
            "BicycleBell": 1,
            "Blender": 2,
            "Buzzer": 3,
            "Clapping": 4,
            "Cough": 5,
            "CupboardOpenClose": 6,
            "Dishes": 7,
            "Doorbell": 8,
            "FootSteps": 9,
            "HairDryer": 10,
            "MechanicalFans": 11,
            "MusicalKeyboard": 12,
            "Percussion": 13,
            "Pour": 14,
            "Speech": 15,
            "Typing": 16,
            "VacuumCleaner": 17,
        }


ALL_MAPPINGS = [
    DCASE2023Task3,
    DCASE2021Task3,
    DCASE2020Task3,
    DCASE2025Task4,
]

# Denotes a ClassMapping subclass
TClassMapping = TypeVar("TClassMapping", bound="ClassMapping")


def get_class_mapping_from_string(class_mapping: str) -> Type[TClassMapping]:
    """
    Given a string representation of a class mapping type (e.g., `DCASE2023Task3`), return the correct ClassMapping
    """
    acceptable_values = [t.__name__ for t in ALL_MAPPINGS]
    acceptable_values_upp = [av.upper() for av in acceptable_values]

    if class_mapping.upper() not in acceptable_values_upp:
        raise ValueError(
            f"Cannot find class mapping {class_mapping}: expected one of {', '.join(acceptable_values)}"
        )
    else:
        # Using `next` avoids having to build the whole list
        return next(
            ws for ws in ALL_MAPPINGS if ws.__name__.upper() == class_mapping.upper()
        )


# noinspection PyCallingNonCallable
def sanitize_class_mapping(
    class_mapping: Optional[Union[TClassMapping, dict, str]]
) -> Type[TClassMapping]:
    """
    Sanitizes any class mapping into the correct 'ClassMapping' class.

    Returns:
        Type['ClassMapping']: the sanitized and initialised ClassMapping
    """

    if class_mapping is None:
        return None

    elif isinstance(class_mapping, str):
        # Convert name to mapping class and instantiate it
        return get_class_mapping_from_string(class_mapping)()

    elif isinstance(class_mapping, dict):
        # Build mapping dynamically
        return ClassMapping.from_dict(class_mapping)

    elif isinstance(class_mapping, ClassMapping):
        # Already an instance
        return class_mapping

    elif isinstance(class_mapping, type) and issubclass(class_mapping, ClassMapping):
        # Provided the class, instantiate it
        return class_mapping()

    else:
        raise TypeError(
            f"Could not parse class mapping with type {type(class_mapping)}"
        )
