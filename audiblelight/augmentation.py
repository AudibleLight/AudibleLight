#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides classes and functions for handling spatial and non-spatial audio augmentations.

# Non-spatial augmentations

These classes are wrappers for a variety of external audio augmentations, including:
    - https://spotify.github.io/pedalboard/ (majority of FX as of v.0.9.17, with obvious exemptions e.g. Convolution)
    - https://docs.pytorch.org/audio/main/transforms.html
    - https://librosa.org/doc/main/effects.html

The purpose of wrapping these augmentations, rather than using them directly, is to provide a unified interface. For
every augmentation class, parameters can either be sampled randomly from acceptable default distributions, or provided
by the user. The exact parameters of the FX can then be reconstructed later from the `params` dictionary. Additionally,
some FX (`MultibandEqualizer`, `TimeWarpXXXX`) are newly implemented for AudibleLight.
"""

import math
from random import random
from typing import Any, Callable, Iterator, Optional, Type, Union

import librosa
import numpy as np
from deepdiff import DeepDiff
from pedalboard import time_stretch
from scipy import stats

from audiblelight import config, custom_types, utils


def _identity(input_array: np.ndarray, *_, **__) -> np.ndarray:
    return input_array


class Augmentation:
    """
    Base class for all augmentation objects to inherit from.

    Augmentation objects need to inherit from this base class so that their parameters can be sampled from a
    distribution every time they are called. This distribution can either be user-defined or set according to a
    default. The user can also pass a numeric override for a parameter, in which case it will always be used.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.

    Properties:
        fx (Callable): the callable function applied to the audio.
        params (dict): the arguments passed to `fx`. Will be serialised inside `to_json`.

    """

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
    ):
        self.sample_rate = utils.sanitise_positive_number(sample_rate, cast_to=int)
        self.fx: Union[Callable, list[Callable]] = _identity
        self.params = dict()

    @staticmethod
    def sample_value(
        override: Optional[Union[custom_types.Numeric, custom_types.DistributionLike]],
        default_dist: custom_types.DistributionLike,
    ) -> custom_types.Numeric:
        """
        Samples a value according to the following method:
            - If override is not provided, a value will be sampled from the `default_dist` distribution.
            - If the override is numeric, it will be used.
            - If the override is a distribution, it will be sampled from the `default_dist` distribution.
            - Otherwise, an error will be raised
        """
        # No override, use default distribution
        if override is None:
            return utils.sanitise_distribution(default_dist).rvs()

        # Override is numeric, use this
        elif isinstance(override, custom_types.Numeric):
            return override

        else:
            # Override is a distribution
            try:
                return utils.sanitise_distribution(override).rvs()

            # We don't know what the distribution is
            except TypeError:
                raise TypeError(f"Cannot handle type {type(override)}")

    def process(self, input_array: np.ndarray) -> np.ndarray:
        """
        Calls the underlying FX (or a list of FX)

        Arguments:
            input_array (np.ndarray): input audio array

        Returns:
            np.ndarray: processed audio array
        """

        # Make a copy so we don't alter the underlying audio
        out = input_array.copy()

        # Process all the FX in sequence
        for fx in self.fx if isinstance(self.fx, list) else [self.fx]:
            out = fx(
                out,
                sample_rate=self.sample_rate,
                buffer_size=config.BUFFER_SIZE,
                reset=True,
            )

        # Temporary convert mono to stereo for pad function
        if out.ndim == 1:
            out = np.expand_dims(out, 0)

        # Pad or truncate the audio to keep the same dims
        #  When padding, use wrap to take samples from the start and put them at the end
        #  E.g., if speeding up audio, wrap the start of the audio back around to the end
        trunc = utils.pad_or_truncate_audio(
            out, max(input_array.shape), pad_mode="wrap"
        )

        # Stereo input, stereo output
        if input_array.ndim == 2:
            return trunc
        # Mono input, mono output
        else:
            return trunc[0, :]

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """
        Alias for `self.process`.
        """
        return self.process(input_array)

    def __repr__(self) -> str:
        """
        Dumps a prettified representation of the parameters used in the FX object
        """
        return utils.repr_as_json(self)

    def __str__(self) -> str:
        """
        Returns a string representation of the augmentation
        """
        combined_args = ", ".join(f"{k}: {v}" for k, v in self.params.items())
        return f"Augmentation '{self.name}' with parameters {combined_args}"

    def __len__(self) -> int:
        """
        Returns the number of FX in this augmentation
        """
        return 1 if not isinstance(self.fx, list) else len(self.fx)

    def __iter__(self) -> Iterator[Callable]:
        """
        Yields an iterator of Event objects from the current scene
        """
        fx_list = self.fx if isinstance(self.fx, list) else [self.fx]
        yield from fx_list

    def to_dict(self) -> dict:
        """
        Returns the parameters used by this augmentation
        """
        return dict(
            name=self.name,
            sample_rate=self.sample_rate,
            **self.params,
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> Type["Augmentation"]:
        """
        Initialise an augmentation from a dictionary.

        Note that the returned object will not be of the `Augmentation` type, but one of its child classes. So,
        attempting to initialise a dictionary where "name" == "LowpassFilter" will instead return a `LowpassFilter`
        object, not an `Augmentation` object.

        Arguments:
            input_dict: Dictionary that will be used to instantiate the object.

        Returns:
            Augmentation child class instance.
        """

        if "name" not in input_dict:
            raise KeyError("Augmentation name must be specified in dictionary")

        # Try and grab the augmentation class based on its name
        augment_name = input_dict["name"]
        try:
            augment_cls = globals()[augment_name]
        except KeyError:
            raise KeyError(f"Augmentation class {augment_name} not found")

        # Check that the remaining kwargs are valid for this class
        input_dict.pop("name")
        utils.validate_kwargs(augment_cls.__init__, **input_dict)

        # Initialise the class with the arguments
        return augment_cls(**input_dict)

    def __eq__(self, other: Any) -> bool:
        """
        Compare two Augmentation objects for equality.

        Internally, we convert both objects to a dictionary, and then use the `deepdiff` package to compare them, with
        some additional logic to account e.g. for significant digits and values that will always be different (e.g.,
        creation time).

        Arguments:
            other: the object to compare the current `Augmentation` object against

        Returns:
            bool: True if the Augmentation objects are equivalent, False otherwise
        """

        # Non-Augmentation objects are always not equal
        if not issubclass(type(other), Augmentation):
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

    @property
    def name(self) -> str:
        """
        Returns the name of this augmentation
        """
        return type(self).__name__


class EventAugmentation(Augmentation):
    """
    Base class for all Augmentation objects that can be used with Events
    """

    AUGMENTATION_TYPE = "event"


class SceneAugmentation(Augmentation):
    """
    Base class for all Augmentation objects that can be used with Scenes
    """

    AUGMENTATION_TYPE = "scene"


class Bitcrush(EventAugmentation):
    """
    Applies a bitcrush effect to the audio input.

    Bitcrushing quantizes the "vertical" resolution of the audio input, such that every sample can only take a certain
    number of unique values, controlled by the bit depth.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        bit_depth: the bit depth to quantize the signal to; will be sampled between 8 and 32 bits if not provided.
    """

    MIN_DEPTH, MAX_DEPTH = 8, 32

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        bit_depth: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )
        self.bit_depth = utils.sanitise_positive_number(
            self.sample_value(
                bit_depth,
                stats.uniform(self.MIN_DEPTH, self.MAX_DEPTH - self.MIN_DEPTH),
            )
        )
        self.params = dict(bit_depth=self.bit_depth)

        from pedalboard import Bitcrush as PBBitcrush

        self.fx = PBBitcrush(**self.params)


class LowpassFilter(EventAugmentation):
    """
    Applies a low-pass filter to the audio.

    By default, the cutoff frequency for the filter will be sampled randomly between 5512 and 22050 Hz. Either the
    exact cutoff frequency or a distribution to sample this from can be provided as arguments to the function.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        cutoff_frequency_hz (Union[custom_types.Numeric, custom_custom_types.DistributionLike]): the cutoff frequency for the filter, or a
            distribution-like object to sample this from. Will default to sampling from uniform distribution
            between 5512 and 22050 Hz if not provided.
    """

    MIN_FREQ, MAX_FREQ = 5512, 22050

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        cutoff_frequency_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        # Initialise the parent class
        super().__init__(
            sample_rate,
        )

        # Handle sampling the cutoff frequency
        #  Can be a numeric value or a distribution passed by the user
        #  Sampled from this default distribution if not given
        self.cutoff_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                cutoff_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )
        self.params = dict(cutoff_frequency_hz=self.cutoff_frequency_hz)

        # Initialise the FX with the required parameter
        from pedalboard import LowpassFilter as PBLowpassFilter

        self.fx = PBLowpassFilter(**self.params)


class HighShelfFilter(EventAugmentation):
    """
    Applies a high-shelf filter to the audio.

    The high-shelf filter has a variable Q (sharpness) and gain parameter, alongside a cutoff frequency. Frequencies
    above this value will be boosted by the provided gain, given in decibels.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        cutoff_frequency_hz: the cutoff frequency for the filter; will be sampled between 5512 and 22050 Hz if not given
        gain_db: the gain of the filter, in dB; will be sampled between -20 and 10 dB, if not given
        q: the Q (or sharpness) of the filter; will be sampled between 0.1 and 1.0 if not given
    """

    MIN_FREQ, MAX_FREQ = 5512, 22050
    MIN_GAIN, MAX_GAIN = -20, 10
    MIN_Q, MAX_Q = 0.1, 1.0

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        gain_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        cutoff_frequency_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        q: Optional[Union[custom_types.Numeric, custom_types.DistributionLike]] = None,
    ):
        super().__init__(
            sample_rate,
        )

        self.cutoff_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                cutoff_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )
        self.gain_db = self.sample_value(
            gain_db,
            stats.uniform(self.MIN_GAIN, self.MAX_GAIN - self.MIN_GAIN),
        )
        self.q = utils.sanitise_positive_number(
            self.sample_value(
                q,
                stats.uniform(self.MIN_Q, self.MAX_Q - self.MIN_Q),
            )
        )
        self.params = dict(
            cutoff_frequency_hz=self.cutoff_frequency_hz, gain_db=self.gain_db, q=self.q
        )

        from pedalboard import HighShelfFilter as PBFilter

        self.fx = PBFilter(**self.params)


class HighpassFilter(EventAugmentation):
    """
    Applies a high-pass filter to the audio.

    By default, the cutoff frequency for the filter will be sampled randomly between 32 and 1024 Hz. Either the
    exact cutoff frequency or a distribution to sample this from can be provided as arguments to the function.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        cutoff_frequency_hz (Union[custom_types.Numeric, custom_custom_types.DistributionLike]): the cutoff frequency for the filter, or a
            distribution-like object to sample this from. Will default to sampling from uniform distribution
            between 32 and 1024 Hz if not provided.
    """

    MIN_FREQ, MAX_FREQ = 32, 1024

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        cutoff_frequency_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        # Initialise the parent class
        super().__init__(sample_rate)

        # Handle sampling the cutoff frequency
        #  Can be a numeric value or a distribution passed by the user
        #  Sampled from this default distribution if not given
        self.cutoff_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                cutoff_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )
        self.params = dict(cutoff_frequency_hz=self.cutoff_frequency_hz)

        # Initialise the FX with the required parameter
        from pedalboard import HighpassFilter as PBHighpassFilter

        self.fx = PBHighpassFilter(**self.params)


class LowShelfFilter(EventAugmentation):
    """
    Applies a low-shelf filter to the audio.

    The low-shelf filter has a variable Q (sharpness) and gain parameter, alongside a cutoff frequency. Frequencies
    below this value will be boosted by the provided gain, given in decibels.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        cutoff_frequency_hz: the cutoff frequency for the filter; will be sampled between 32 and 1024 Hz if not given
        gain_db: the gain of the filter, in dB; will be sampled between -20 and 10 dB, if not given
        q: the Q (or sharpness) of the filter; will be sampled between 0.1 and 1.0 if not given
    """

    MIN_FREQ, MAX_FREQ = 32, 1024
    MIN_GAIN, MAX_GAIN = -20, 10
    MIN_Q, MAX_Q = 0.1, 1.0

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        gain_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        cutoff_frequency_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        q: Optional[Union[custom_types.Numeric, custom_types.DistributionLike]] = None,
    ):
        super().__init__(
            sample_rate,
        )

        self.cutoff_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                cutoff_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )
        self.gain_db = self.sample_value(
            gain_db,
            stats.uniform(self.MIN_GAIN, self.MAX_GAIN - self.MIN_GAIN),
        )
        self.q = utils.sanitise_positive_number(
            self.sample_value(
                q,
                stats.uniform(self.MIN_Q, self.MAX_Q - self.MIN_Q),
            )
        )
        self.params = dict(
            cutoff_frequency_hz=self.cutoff_frequency_hz, gain_db=self.gain_db, q=self.q
        )

        from pedalboard import LowShelfFilter as PBFilter

        self.fx = PBFilter(**self.params)


class MultibandEqualizer(EventAugmentation):
    """
    Applies equalization to the audio.

    The Equalizer applies N peak filter objects to the audio. The gain, frequency, and Q of each filter can be
    set independently, or randomised from within an acceptable range. By default, between 1 and 8 individual peak
    filters are applied, with randomly selected gain, frequency, and Q of each filter.

    Using this class, it is possible to create a multiband equalizer similar to the "parametric EQ" plugins often
    featured in digital audio workstations. For additional flexibility, consider combining with both `HighpassFilter`
    and `LowpassFilter`.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        n_bands: the number of peak filters to use in the equalizer. Defaults to a random integer between 1 and 8.
        gain_db: the gain values for each peak. Can be either a single value, a list of N values, or a distribution.
            If a single value, will be repeated N times. If a distribution, will be sampled from N times.
        cutoff_frequency_hz: the frequency for each peak filter. Same rules as `gain_db` apply.
        q: the "sharpness" of each filter. Same rules as `gain_db` apply.
    """

    MIN_BANDS, MAX_BANDS = 1, 8
    MIN_GAIN, MAX_GAIN = -20, 10
    MIN_FREQ, MAX_FREQ = 1024, 22050
    MIN_Q, MAX_Q = 0.1, 1.0

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        n_bands: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        gain_db: Optional[
            Union[
                list[custom_types.Numeric],
                custom_types.Numeric,
                custom_types.DistributionLike,
            ]
        ] = None,
        cutoff_frequency_hz: Optional[
            Union[
                list[custom_types.Numeric],
                custom_types.Numeric,
                custom_types.DistributionLike,
            ]
        ] = None,
        q: Optional[
            Union[
                list[custom_types.Numeric],
                custom_types.Numeric,
                custom_types.DistributionLike,
            ]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )

        # The number of frequency bands we'll be applying
        self.n_bands = utils.sanitise_positive_number(
            self.sample_value(
                n_bands,
                stats.uniform(self.MIN_BANDS, self.MAX_BANDS - self.MIN_BANDS),
            ),
            cast_to=int,
        )

        # Sample the parameters for all N frequency bands
        self.gain_db = self.sample_peak_filter_params(
            gain_db, stats.uniform(self.MIN_GAIN, self.MAX_GAIN - self.MIN_GAIN)
        )
        self.cutoff_frequency_hz = self.sample_peak_filter_params(
            cutoff_frequency_hz,
            stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
        )
        self.q = self.sample_peak_filter_params(
            q, stats.uniform(self.MIN_Q, self.MAX_Q - self.MIN_Q)
        )
        self.params = dict(
            n_bands=self.n_bands,
            gain_db=self.gain_db,
            cutoff_frequency_hz=self.cutoff_frequency_hz,
            q=self.q,
        )

        # Given the parameter settings, create the filters
        self.fx = self.create_filters()

    # noinspection PyUnreachableCode,PyUnresolvedReferences
    def sample_peak_filter_params(
        self,
        override: Union[
            custom_types.Numeric,
            list[custom_types.Numeric],
            custom_types.DistributionLike,
        ],
        default_dist: custom_types.DistributionLike,
    ) -> list[custom_types.Numeric]:
        """
        Samples all values (e.g., all Q values, all frequencies) for all N peak filters.

        Uses the following method:
            - If override not provided, sample from default_dist N times
            - If override provided and is a list or iterable, use this
            - If override provided and is numeric, use this N times (repeated)
            - If override provided and is a distribution, sample from this N times
            - Otherwise, raise an error
        """

        # No override provided: sample N times from default distribution
        if override is None:
            default_dist = utils.sanitise_distribution(default_dist)
            return [default_dist.rvs() for _ in range(self.n_bands)]

        # Override is a list: check that it is the correct length and return
        elif isinstance(override, (list, np.ndarray)):
            if len(override) != self.n_bands:
                raise ValueError(
                    f"Expected {self.n_bands} values but got {len(override)}"
                )
            return override if isinstance(override, list) else override.tolist()

        # Override is a single value: return this value N times
        elif isinstance(override, custom_types.Numeric):
            return [override for _ in range(self.n_bands)]

        else:
            # Override is a distribution
            try:
                dist = utils.sanitise_distribution(override)
                return [dist.rvs() for _ in range(self.n_bands)]

            # We don't know what override is
            except TypeError:
                raise TypeError(f"Cannot handle type {type(override)}")

    def create_filters(self) -> list[Callable]:
        """
        Creates multiple `PeakFilter` effects with given gain, frequency and q values
        """
        from pedalboard import PeakFilter

        filters = []
        for gain, freq, q in zip(self.gain_db, self.cutoff_frequency_hz, self.q):
            # Create the filter
            filters.append(
                PeakFilter(
                    cutoff_frequency_hz=utils.sanitise_positive_number(freq),
                    gain_db=gain,
                    q=utils.sanitise_positive_number(q),
                )
            )

        return filters


class Compressor(EventAugmentation):
    """
    Applies compression to the audio signal.

    A dynamic range compressor, used to reduce the volume of loud sounds and "compress" the loudness of the signal. For
    a lossy compression algorithm that introduces noise or artifacts, see `MP3Compressor` or `GSMCompressor`.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        threshold_db: the dB threshold after which the compressor is active. Sampled between -40 and -20 dB if not given
        ratio: the compressor ratio, i.e. `ratio=4` reduces the signal volume by 4 dB for every 1 dB over the threshold.
            If not provided, sampled from [4, 8, 12, 20] (i.e., the ratio values on the famous UREI 1176 compressor)
        attack_ms: the time taken for the compressor to kick in after the threshold is exceeded. If not provided,
            will be sampled between 1 and 100 ms.
        release_ms: the time taken for the compressor to return to 0 dB after exceeding the threshold. If not provided,
            will be sampled between 50 and 1100 ms (again, inspired by the UREI 1176).
    """

    # The ratio values here are taken from the famous UREI 1176 compressor
    RATIOS = [4, 8, 12, 20]
    MIN_THRESHOLD_DB, MAX_THRESHOLD_DB = -40, -20
    MIN_ATTACK, MAX_ATTACK = 1, 100
    MIN_RELEASE, MAX_RELEASE = 50, 1100

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        threshold_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        ratio: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        attack_ms: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        release_ms: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)

        # Set all FX parameters
        self.threshold_db = int(
            (
                self.sample_value(
                    threshold_db,
                    stats.uniform(self.MIN_THRESHOLD_DB, abs(self.MAX_THRESHOLD_DB)),
                )
            )
        )
        # Threshold dB value should always be negative
        self.threshold_db = -abs(self.threshold_db)

        self.ratio = int(
            utils.sanitise_positive_number(
                self.sample_value(ratio, lambda: np.random.choice(self.RATIOS))
            )
        )
        self.attack_ms = utils.sanitise_positive_number(
            self.sample_value(
                attack_ms,
                stats.uniform(self.MIN_ATTACK, self.MAX_ATTACK - self.MIN_ATTACK),
            )
        )
        self.release_ms = utils.sanitise_positive_number(
            self.sample_value(
                release_ms,
                stats.uniform(self.MIN_RELEASE, self.MAX_RELEASE - self.MIN_RELEASE),
            )
        )
        self.params = dict(
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
        )

        from pedalboard import Compressor as PBCompressor

        self.fx = PBCompressor(**self.params)


class Chorus(EventAugmentation):
    """
    Applies chorus to the audio.

    This audio effect can be controlled via the speed and depth of the LFO controlling the frequency response,
    a mix control, a feedback control, and the centre delay of the modulation.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        rate_hz: the speed of the LFO controlling the frequency response. By default, sampled between 0 and 10 Hz
        depth: the depth of the LFO controlling the frequency response. By default, sampled between 0 and 1.0.
        centre_delay_ms: the centre delay of the modulation. By default, sampled between 1 and 20 ms.
        feedback: the feedback of the effect. By default, sampled between 0.0 and 0.9.
        mix: the dry/wet mix of the effect. By default, sampled between 0.1 and 0.5.
    """

    MIN_RATE, MAX_RATE = 0, 10
    MIN_DEPTH, MAX_DEPTH = 0.0, 1.0
    MIN_DELAY, MAX_DELAY = 1.0, 20.0
    MIN_MIX, MAX_MIX = 0.1, 0.5
    MIN_FEEDBACK, MAX_FEEDBACK = 0.0, 0.9

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        rate_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        depth: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        centre_delay_ms: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        feedback: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        mix: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )

        # Initialise all the FX parameters
        self.rate_hz = utils.sanitise_positive_number(
            self.sample_value(
                rate_hz, stats.uniform(self.MIN_RATE, self.MAX_RATE - self.MIN_RATE)
            )
        )
        self.depth = utils.sanitise_positive_number(
            self.sample_value(
                depth, stats.uniform(self.MIN_DEPTH, self.MAX_DEPTH - self.MIN_DEPTH)
            )
        )
        self.centre_delay_ms = utils.sanitise_positive_number(
            self.sample_value(
                centre_delay_ms,
                stats.uniform(self.MIN_DELAY, self.MAX_DELAY - self.MIN_DELAY),
            )
        )
        self.feedback = utils.sanitise_positive_number(
            self.sample_value(
                feedback,
                stats.uniform(self.MIN_FEEDBACK, self.MAX_FEEDBACK - self.MIN_FEEDBACK),
            )
        )
        self.mix = utils.sanitise_positive_number(
            self.sample_value(
                mix, stats.uniform(self.MIN_MIX, self.MAX_MIX - self.MIN_MIX)
            )
        )
        self.params = dict(
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_delay_ms=self.centre_delay_ms,
            feedback=self.feedback,
            mix=self.mix,
        )

        from pedalboard import Chorus as PBChorus

        self.fx = PBChorus(**self.params)


class Clipping(EventAugmentation):
    """
    Applies hard distortion to the audio.

    Clips the audio signal at the provided threshold, in decibels.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        threshold_db: the dB level of the distortion effect. By default, will be sampled between -10 and -1 dB.
    """

    MIN_THRESHOLD_DB, MAX_THRESHOLD_DB = -10, -1

    def __init__(
        self,
        sample_rate: custom_types.Numeric = config.SAMPLE_RATE,
        threshold_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        # Set all FX parameters
        self.threshold_db = int(
            (
                self.sample_value(
                    threshold_db,
                    stats.uniform(self.MIN_THRESHOLD_DB, abs(self.MAX_THRESHOLD_DB)),
                )
            )
        )
        # Threshold dB value should always be negative
        self.threshold_db = -abs(self.threshold_db)
        self.params = dict(threshold_db=self.threshold_db)

        from pedalboard import Clipping as PBClipping

        self.fx = PBClipping(**self.params)


class Limiter(EventAugmentation):
    """
    Applies limiting to the audio.

    A simple limiter with a hard clipper set to 0 dB. Release and threshold dB can be controlled by the user.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        threshold_db: the dB threshold after which the compressor is active. Sampled between -40 and -20 dB if not given
        release_ms: the time taken for the compressor to return to 0 dB after exceeding the threshold. If not provided,
            will be sampled between 50 and 1100 ms (again, inspired by the UREI 1176).
    """

    MIN_THRESHOLD_DB, MAX_THRESHOLD_DB = -40, -20
    MIN_RELEASE, MAX_RELEASE = 50, 1100

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        threshold_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        release_ms: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )

        # Set all FX parameters
        self.threshold_db = int(
            (
                self.sample_value(
                    threshold_db,
                    stats.uniform(self.MIN_THRESHOLD_DB, abs(self.MAX_THRESHOLD_DB)),
                )
            )
        )
        # Threshold dB value should always be negative
        self.threshold_db = -abs(self.threshold_db)

        self.release_ms = utils.sanitise_positive_number(
            self.sample_value(
                release_ms,
                stats.uniform(self.MIN_RELEASE, self.MAX_RELEASE - self.MIN_RELEASE),
            )
        )
        self.params = dict(threshold_db=self.threshold_db, release_ms=self.release_ms)

        # Initialise the audio FX
        from pedalboard import Limiter as PBLimiter

        self.fx = PBLimiter(**self.params)


class Distortion(EventAugmentation):
    """
    Applies distortion to the audio.

    Applies a non-linear (tanh, or hyperbolic tangent) waveshaping function to apply harmonically pleasing distortion
    to a signal.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        drive_db: the dB level of the distortion effect. By default, will be sampled between 10 and 30 dB.
    """

    MIN_DRIVE, MAX_DRIVE = 10, 30

    def __init__(
        self,
        sample_rate: custom_types.Numeric = config.SAMPLE_RATE,
        drive_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )
        self.drive_db = utils.sanitise_positive_number(
            self.sample_value(
                drive_db, stats.uniform(self.MIN_DRIVE, self.MAX_DRIVE - self.MIN_DRIVE)
            )
        )
        self.params = dict(drive_db=self.drive_db)

        from pedalboard import Distortion as PBDistortion

        self.fx = PBDistortion(drive_db=self.drive_db)


class Phaser(EventAugmentation):
    """
    Applies a phaser to the audio.

    A 6 stage phaser that modulates first order all-pass filters to create sweeping notches in the magnitude frequency
    response. This audio effect can be controlled with standard phaser parameters: the speed and depth of the LFO
    controlling the frequency response, a mix control, a feedback control, and the centre frequency of the modulation.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        rate_hz: the speed of the LFO controlling the frequency response. By default, sampled between 0 and 10 Hz
        depth: the depth of the LFO controlling the frequency response. By default, sampled between 0 and 1.0.
        centre_frequency_hz: the centre frequency of the modulation. By default, sampled between 1 and 20 ms.
        feedback: the feedback of the effect. By default, sampled between 0.0 and 0.9.
        mix: the dry/wet mix of the effect. By default, sampled between 0.1 and 0.5.
    """

    MIN_RATE, MAX_RATE = 0, 10
    MIN_DEPTH, MAX_DEPTH = 0.0, 1.0
    MIN_FREQ, MAX_FREQ = 260, 6500
    MIN_MIX, MAX_MIX = 0.1, 0.5
    MIN_FEEDBACK, MAX_FEEDBACK = 0.0, 0.9

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        rate_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        depth: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        centre_frequency_hz: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        feedback: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        mix: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        self.rate_hz = utils.sanitise_positive_number(
            self.sample_value(
                rate_hz, stats.uniform(self.MIN_RATE, self.MAX_RATE - self.MIN_RATE)
            )
        )
        self.depth = utils.sanitise_positive_number(
            self.sample_value(
                depth, stats.uniform(self.MIN_DEPTH, self.MAX_DEPTH - self.MIN_DEPTH)
            )
        )
        self.centre_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                centre_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )
        self.feedback = utils.sanitise_positive_number(
            self.sample_value(
                feedback,
                stats.uniform(self.MIN_FEEDBACK, self.MAX_FEEDBACK - self.MIN_FEEDBACK),
            )
        )
        self.mix = utils.sanitise_positive_number(
            self.sample_value(
                mix, stats.uniform(self.MIN_MIX, self.MAX_MIX - self.MIN_MIX)
            )
        )
        self.params = dict(
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_frequency_hz=self.centre_frequency_hz,
            feedback=self.feedback,
            mix=self.mix,
        )

        from pedalboard import Phaser as PBPhaser

        self.fx = PBPhaser(**self.params)


class Delay(EventAugmentation):
    """
    Applies delay to the audio.

    A digital delay plugin with controllable delay time, feedback percentage, and dry/wet mix.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        delay_seconds: the delay time for the effect, in seconds. By default, sampled between 0.01 and 1.0 seconds.
        feedback: the feedback of the effect. By default, sampled between 0.1 and 0.5.
        mix: the dry/wet mix of the effect. By default, sampled between 0.1 and 0.5
    """

    MIN_DELAY, MAX_DELAY = 0.01, 1.0
    MIN_FEEDBACK, MAX_FEEDBACK = 0.1, 0.5
    MIN_MIX, MAX_MIX = 0.1, 0.5

    def __init__(
        self,
        sample_rate: custom_types.Numeric = config.SAMPLE_RATE,
        delay_seconds: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        feedback: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        mix: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        self.delay_seconds = utils.sanitise_positive_number(
            self.sample_value(
                delay_seconds,
                stats.uniform(self.MIN_DELAY, self.MAX_DELAY - self.MIN_DELAY),
            )
        )
        self.feedback = utils.sanitise_positive_number(
            self.sample_value(
                feedback,
                stats.uniform(self.MIN_FEEDBACK, self.MAX_FEEDBACK - self.MIN_FEEDBACK),
            )
        )
        self.mix = utils.sanitise_positive_number(
            self.sample_value(
                mix, stats.uniform(self.MIN_MIX, self.MAX_MIX - self.MIN_MIX)
            )
        )
        self.params = dict(
            delay_seconds=self.delay_seconds,
            feedback=self.feedback,
            mix=self.mix,
        )

        from pedalboard import Delay as PBDelay

        self.fx = PBDelay(**self.params)


class Gain(EventAugmentation):
    """
    Applies gain (volume) to the audio.

    A gain plugin that increases or decreases the volume of a signal by amplifying or attenuating it by the
    provided value (in decibels). No distortion or other effects are applied.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        gain_db: the gain to apply to the signal. By default, sampled between -10 and 10 dB.
    """

    MIN_GAIN, MAX_GAIN = -10, 10

    def __init__(
        self,
        sample_rate: custom_types.Numeric = config.SAMPLE_RATE,
        gain_db: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )
        self.gain_db = self.sample_value(
            gain_db, stats.uniform(self.MIN_GAIN, self.MAX_GAIN - self.MIN_GAIN)
        )
        self.params = dict(gain_db=self.gain_db)

        from pedalboard import Gain as PBGain

        self.fx = PBGain(**self.params)


class GSMFullRateCompressor(EventAugmentation):
    """
    Applies GSM compression to the audio.

    An audio degradation/compression plugin that applies the GSM “Full Rate” compression algorithm to emulate the
    sound of a 2G cellular phone connection. This plugin internally resamples the input audio to a fixed sample rate
    of 8kHz (required by the GSM Full Rate codec), although the quality of the resampling algorithm can be specified.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        quality: the quality of the resampling. By default, will be sampled between 0 and 3 (inclusive).
    """

    # Don't use the highest resampling quality (4) as it is much slower than the others
    QUALITIES = range(4)

    def __init__(
        self,
        sample_rate: custom_types.Numeric = config.SAMPLE_RATE,
        quality: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        self.quality = int(
            utils.sanitise_positive_number(
                self.sample_value(quality, lambda: np.random.choice(self.QUALITIES))
            )
        )
        self.params = dict(quality=self.quality)

        from pedalboard import GSMFullRateCompressor as PBGSMFullRateCompressor
        from pedalboard import Resample

        self.fx = PBGSMFullRateCompressor(quality=Resample.Quality(self.quality))


class MP3Compressor(EventAugmentation):
    """
    Applies the LAME MP3 encoder in real-time to add compression artifacts to the audio stream.

    Currently only supports variable bit-rate mode (VBR) and accepts a floating-point VBR quality value
    (between 0.0 and 10.0; lower is better). Note that the MP3 format only supports 8kHz, 11025Hz, 12kHz, 16kHz,
    22050Hz, 24kHz, 32kHz, 44.1kHz, and 48kHz audio; if an unsupported sample rate is provided, an exception will be
    thrown at processing time.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        vbr_quality: the quality of the resampling. By default, will be sampled between 2 and 10.
    """

    VBR_MIN, VBR_MAX = 2.001, 9.999
    SUPPORTED_SAMPLE_RATES = [
        8000,
        11025,
        12000,
        16000,
        22050,
        24000,
        32000,
        44100,
        48000,
    ]

    def __init__(
        self,
        sample_rate: custom_types.Numeric = config.SAMPLE_RATE,
        vbr_quality: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)

        # Explicitly raise an error if the sample rate is not supported
        if self.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            supporteds = " Hz, ".join([str(i) for i in self.SUPPORTED_SAMPLE_RATES])
            raise ValueError(
                f"Expected sample rate to be one of {supporteds}, but got {self.sample_rate}"
            )

        # Sample VBR quality and construct parameters dictionary
        self.vbr_quality = utils.sanitise_positive_number(
            self.sample_value(
                vbr_quality, stats.uniform(self.VBR_MIN, self.VBR_MAX - self.VBR_MIN)
            )
        )
        self.params = dict(vbr_quality=self.vbr_quality)

        from pedalboard import MP3Compressor as PBMP3Compressor

        self.fx = PBMP3Compressor(**self.params)


class PitchShift(EventAugmentation):
    """
    Applies pitch-shifting to the audio.

    Internally, this function uses the `time_shift` function in pedalboard, specifying the `pitch_shift` as an array
    with the same dims as `input_audio`. From our initial testing, we found this algorithm to be significantly faster
    than many other existing pitch shifting algorithms, including `librosa` and `pyrubberband`.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        semitones: the number of semitones to shift the audio by. By default, will be sampled from between +/- 3
            semitones (i.e., up or down a minor third).
    """

    MIN_SEMITONES, MAX_SEMITONES = -3, 3

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        semitones: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(
            sample_rate,
        )

        self.semitones = int(
            self.sample_value(
                semitones,
                stats.uniform(
                    self.MIN_SEMITONES, self.MAX_SEMITONES - self.MIN_SEMITONES
                ),
            )
        )
        self.fx = self._apply_fx
        self.params = dict(semitones=self.semitones)

    def _apply_fx(self, input_array: np.ndarray, *_, **__) -> np.ndarray:
        """
        Little hack, given that `Pedalboard.time_stretch` is a function, not a class
        """
        return time_stretch(
            input_array,
            samplerate=self.sample_rate,
            stretch_factor=1.0,
            pitch_shift_in_semitones=self.semitones
            * np.ones_like(input_array),  # Nasty little speedup...
            high_quality=False,
        )

    def process(self, input_array: np.ndarray) -> np.ndarray:
        """
        Apply the effect to the input audio.
        """
        # Simply return the input if we're not using pitch shifting
        if self.semitones == 0:
            return input_array
        return super().process(input_array)


class SpeedUp(EventAugmentation):
    """
    Changes the speed of the audio.

    Using a higher stretch_factor will shorten the audio - i.e., a stretch_factor of 2.0 will double the speed of the
    audio and halve the length of the audio, without changing the pitch of the audio. When the output audio is shorter
    than the input, it will be right-padded with zeros to maintain the correct dim. When the output audio is longer
    than the input, it will be truncated to maintain the correct dim.

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        stretch_factor: the time-stretching factor to apply. Values above 1 will increase the speed of the audio, while
            values below 1 will decrease the speed. A value of 1 will have no effect. By default, will be sampled
            from between 0.7 and 1.5.
    """

    MIN_SHIFT, MAX_SHIFT = 0.7, 1.5

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        stretch_factor: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        self.stretch_factor = utils.sanitise_positive_number(
            self.sample_value(
                stretch_factor,
                stats.uniform(self.MIN_SHIFT, self.MAX_SHIFT - self.MIN_SHIFT),
            )
        )
        self.fx = self._apply_fx
        self.params = dict(stretch_factor=self.stretch_factor)

    def _apply_fx(self, input_array: np.ndarray, *_, **__) -> np.ndarray:
        """
        Little hack, given that `Pedalboard.time_stretch` is a function, not a class
        """
        return time_stretch(
            input_array,
            samplerate=self.sample_rate,
            stretch_factor=self.stretch_factor,
            pitch_shift_in_semitones=0.0,
            high_quality=False,
        )

    def process(self, input_array: np.ndarray) -> np.ndarray:
        """
        Apply the effect to the input audio.
        """
        # Identity operation
        if self.stretch_factor == 1.0:
            return input_array
        return super().process(input_array)


class Preemphasis(EventAugmentation):
    r"""
    Applies preemphasis to the audio.

    Pre-emphasizes an audio signal with a first-order differencing filter, such that

    ..math::
        y[n] = y[n] - \text{coef} \times y[n-1], \ \text{where coef} \in \{0, 1\}

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        coef: the coefficient for pre-emphasis, sampled between 0 and 1 when not provided.
            At `coef=0`, the signal is unchanged. At `coef=1`, the result is the first-order difference of the signal.
    """

    MIN_COEF, MAX_COEF = 0.0, 1.0

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        coef: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        self.coef = utils.sanitise_positive_number(
            self.sample_value(
                coef,
                stats.uniform(self.MIN_COEF, self.MAX_COEF - self.MIN_COEF),
            )
        )
        self.fx = self._apply_fx
        self.params = dict(coef=self.coef)

    def _apply_fx(self, input_audio: np.ndarray, *_, **__) -> np.ndarray:
        return librosa.effects.preemphasis(input_audio, coef=self.coef)


class Deemphasis(Preemphasis):
    r"""
    Applies deemphasis to the audio.

    De-emphasizes an audio signal with the inverse operation to preemphasis, such that

    ..math::
        y[n] = f(y)[n] + \text{coef} \times y[n-1], \\
        \text{where coef} \in \{0, 1\} \ \text{and} \ f = \texttt{preemphasis}(y, \text{coef})}
    """

    def _apply_fx(self, input_audio: np.ndarray, *_, **__) -> np.ndarray:
        return librosa.effects.deemphasis(input_audio, coef=self.coef)


class Fade(EventAugmentation):
    """
    Add a fade-in and/or fade-out to audio.

    The shape of the fade can be specified as one of:
        - "linear"
        - "exponential"
        - "logarithmic"
        - "quarter_sine"
        - "half_sine"
        - "none", equivalent to no fade

    Different shapes can be specified for both the fade-in and fade-out. The length of the fade-in and fade-out time
    must be specified in seconds.

    For more information:
        https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.Fade.html

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        fade_in_len: the length of time for the fade in (seconds), sampled between 0 and 1 if not given.
        fade_out_len: the length of time for the fade out (seconds), sampled between 0 and 1 if not given.
        fade_in_shape: the shape of the fade in, sampled randomly from an available option (see above) if not given
        fade_out_shape: the shape of the fade out, sampled randomly from an available option (see above) if not given
    """

    MIN_FADE, MAX_FADE = 0.0, 1.0  # seconds
    FADE_SHAPES = [
        "linear",
        "exponential",
        "logarithmic",
        "quarter_sine",
        "half_sine",
        "none",
    ]

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        fade_in_len: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        fade_out_len: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        fade_in_shape: Optional[str] = None,
        fade_out_shape: Optional[str] = None,
    ):
        super().__init__(sample_rate)

        # Get fade-in and fade-out times (in seconds)
        self.fade_in_len = utils.sanitise_positive_number(
            self.sample_value(
                fade_in_len,
                stats.uniform(self.MIN_FADE, self.MAX_FADE - self.MIN_FADE),
            )
        )
        self.fade_out_len = utils.sanitise_positive_number(
            self.sample_value(
                fade_out_len,
                stats.uniform(self.MIN_FADE, self.MAX_FADE - self.MIN_FADE),
            )
        )

        # Get shapes for the fade-in and fade-out
        self.fade_in_shape = self._sample_fade_shape(fade_in_shape)
        self.fade_out_shape = self._sample_fade_shape(fade_out_shape)

        self.fx = self._apply_fx
        self.params = dict(
            fade_in_len=self.fade_in_len,
            fade_out_len=self.fade_out_len,
            fade_in_shape=self.fade_in_shape,
            fade_out_shape=self.fade_out_shape,
        )

    def _sample_fade_shape(self, given_shape: Optional[str] = None) -> str:
        if given_shape is None:
            given_shape = np.random.choice(self.FADE_SHAPES)

        if given_shape not in self.FADE_SHAPES:
            raise ValueError(
                f"Expected `shape` to be one of {', '.join(self.FADE_SHAPES)} but got {given_shape}"
            )

        return given_shape

    def _fade_in(self, waveform_length: int, fade_len: int) -> np.ndarray:
        if fade_len == 0 or self.fade_in_shape == "none":
            return np.ones(waveform_length)

        fade = np.linspace(0, 1, fade_len)
        ones = np.ones(waveform_length - fade_len)

        if self.fade_in_shape == "linear":
            pass
        elif self.fade_in_shape == "exponential":
            fade = np.power(2, (fade - 1)) * fade
        elif self.fade_in_shape == "logarithmic":
            fade = np.log10(0.1 + fade) + 1
        elif self.fade_in_shape == "quarter_sine":
            fade = np.sin(fade * math.pi / 2)
        elif self.fade_in_shape == "half_sine":
            fade = np.sin(fade * math.pi - math.pi / 2) / 2 + 0.5

        return np.clip(np.concatenate((fade, ones)), 0, 1)

    def _fade_out(self, waveform_length: int, fade_len: int) -> np.ndarray:
        if fade_len == 0 or self.fade_out_shape == "none":
            return np.ones(waveform_length)

        fade = np.linspace(0, 1, fade_len)
        ones = np.ones(waveform_length - fade_len)

        if self.fade_out_shape == "linear":
            fade = -fade + 1
        elif self.fade_out_shape == "exponential":
            fade = np.power(2, -fade) * (1 - fade)
        elif self.fade_out_shape == "logarithmic":
            fade = np.log10(1.1 - fade) + 1
        elif self.fade_out_shape == "quarter_sine":
            fade = np.sin(fade * math.pi / 2 + math.pi / 2)
        elif self.fade_out_shape == "half_sine":
            fade = np.sin(fade * math.pi + math.pi / 2) / 2 + 0.5

        return np.clip(np.concatenate((ones, fade)), 0, 1)

    def _apply_fx(self, input_audio: np.ndarray, *_, **__) -> np.ndarray:
        """
        Apply the fade effect.

        Args:
            input_audio (np.ndarray): Audio array of shape (..., time)

        Returns:
            np.ndarray: Audio with fade-in and fade-out applied.
        """
        waveform_length = input_audio.shape[-1]
        fade_in_samples = min(
            int(round(self.fade_in_len * self.sample_rate)), waveform_length
        )
        fade_out_samples = min(
            int(round(self.fade_out_len * self.sample_rate)), waveform_length
        )

        fade_in = self._fade_in(waveform_length, fade_in_samples)
        fade_out = self._fade_out(waveform_length, fade_out_samples)
        fade = fade_in * fade_out

        # Reshape fade to match input_audio dimensions
        fade = fade.reshape((1,) * (input_audio.ndim - 1) + (-1,))
        return input_audio * fade


class Invert(EventAugmentation):
    r"""
    Inverts the phase of an input audio array (i.e., flips it "vertically")

    Applies phase inversion, such that the output audio is equivalent to

    ..math::
        y[n] = -y[n]

    Arguments:
        sample_rate (custom_types.Numeric): not used by this augmentation, but required for compatibility with parent
    """

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
    ):
        super().__init__(sample_rate)
        self.fx = self._apply_fx
        self.params = dict()

    def _apply_fx(self, input_audio: np.ndarray, *_, **__) -> np.ndarray:
        # Equivalent to doing `-array` as a callable
        return np.negative(input_audio)


class Reverse(EventAugmentation):
    """
    Reverses an input audio array (i.e., flips it "horizontally").

    Arguments:
        sample_rate (custom_types.Numeric): not used by this augmentation, but required for compatibility with parent
    """

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
    ):
        super().__init__(sample_rate)
        self.fx = self._apply_fx
        self.params = dict()

    def _apply_fx(self, input_audio: np.ndarray, *_, **__) -> np.ndarray:
        # Flip along the last axis (should be equivalent to samples)
        return np.flip(input_audio, axis=-1)


class TimeWarp(EventAugmentation):
    """
    Parent class for all time-warping augmentations.

    The following augmentations are classed as "time-warping":
        - TimeWarpSilence
        - TimeWarpDuplicate
        - TimeWarpReverse
        - TimeWarpRemove

    They take inspiration from "DJ-style" effects commonly used in hip-hop music production and have been used as
    augmentations in both music sample identification [1] and cover song identification [2].

    Arguments:
        sample_rate (custom_types.Numeric): the sample rate for the effect to use.
        fps: the number of frames-per-second to use. Will be sampled between 2 and 10 FPS if not given.
        prob: the probability of activating the effect for every frame. Will be sampled between 5 and 15% if not given.

    References:
        [1] Cheston, H., Van Balen, J., & Durand, S. (2025). Automatic Identification of Samples in Hip-Hop Music via
        Multi-Loss Training and an Artificial Dataset. arXiv preprint arXiv:2502.06364.
        [2] Yesiler, F. Serrà, J., & Gómez, E. (2020). Accurate and Scalable Version Identification Using
        Musically-Motivated Embeddings, ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and
        Signal Processing (ICASSP). doi:10.1109/ICASSP40776.2020.9053793.
    """

    MIN_PROB, MAX_PROB = 0.05, 0.15
    MIN_FPS, MAX_FPS = 2, 10.0

    def __init__(
        self,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        fps: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
        prob: Optional[
            Union[custom_types.Numeric, custom_types.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate)
        self.fps = utils.sanitise_positive_number(
            self.sample_value(
                fps,
                stats.uniform(self.MIN_FPS, self.MAX_FPS - self.MIN_FPS),
            )
        )
        # Need this check or we'll raise zerodivisionerror later
        if self.fps == 0.0:
            raise ValueError(f"Expected fps to be greater than 0 but got {fps}")

        self.prob = utils.sanitise_positive_number(
            self.sample_value(
                prob,
                stats.uniform(self.MIN_PROB, self.MAX_PROB - self.MIN_PROB),
            )
        )
        self.fx = self._apply_fx
        self.params = dict(fps=self.fps, prob=self.prob)

    def _timewarp(self, sliced_audio_frames: np.ndarray) -> list[np.ndarray]:
        """
        Implements the main time-warping functionality.

        This should operate on a list of audio frames obtained using `slice_frames`.
        """
        # Parent class: just return the list of frames
        return sliced_audio_frames

    def _apply_fx(self, input_audio: np.ndarray, *_, **__) -> np.ndarray:
        """
        Applies the audio FX for the time-warping effect
        """
        # Identity operation
        if self.prob == 0:
            return input_audio

        # Non-overlapping frames use equal frame and hop length
        fl = round(self.sample_rate / self.fps)

        # Try and slice the audio into frames
        if fl > max(input_audio.shape):
            # Too short to slice: just use a single frame
            sliced = np.expand_dims(input_audio, 0)
        else:
            sliced = librosa.util.frame(input_audio, frame_length=fl, hop_length=fl)

        # Apply the FX: let's do the timewarp!
        combframes = self._timewarp(sliced)

        # If we've never triggered the effect, return the original audio
        #  The parent class `process` function will handle padding/truncating
        try:
            return np.concatenate(combframes)
        except ValueError:
            return input_audio


class TimeWarpSilence(TimeWarp):
    """
    Applies a time-warping effect (silence) to the audio.

    This effect uses the following method:
        - Split audio into frames, according to FPS value
        - Iterate over all frames
            - Sample a random value `x`
            - if `x < prob`:
                - Silence the frame (replace with zeros)
    """

    def _timewarp(self, sliced_audio_frames: np.ndarray) -> list[np.ndarray]:
        combframes = []
        # Iterate over all the frames
        for frame in sliced_audio_frames:
            # If we trigger the effect, zero the frame
            if random() < self.prob:
                frame = np.zeros(len(frame))
            combframes.append(frame)
        return combframes


class TimeWarpDuplicate(TimeWarp):
    """
    Applies a time-warping effect (silence) to the audio.

    This effect uses the following method:
        - Split audio into frames, according to FPS value
        - Iterate over all frames
            - Sample a random value `x`
            - if `x < prob`:
                - Duplicate the frame
    """

    def _timewarp(self, sliced_audio_frames: np.ndarray) -> list[np.ndarray]:
        combframes = []
        # Iterate over all the frames
        for frame in sliced_audio_frames:
            # If we trigger the effect, append the frame to the list twice
            if random() < self.prob:
                combframes.append(frame)
            combframes.append(frame)
        return combframes


class TimeWarpRemove(TimeWarp):
    """
    Applies a time-warping effect (silence) to the audio.

    This effect uses the following method:
        - Split audio into frames, according to FPS value
        - Iterate over all frames
            - Sample a random value `x`
            - if `x < prob`:
                - Remove the frame
    """

    def _timewarp(self, sliced_audio_frames: np.ndarray) -> list[np.ndarray]:
        combframes = []
        # Iterate over all the frames
        for frame in sliced_audio_frames:
            # If we trigger the effect, skip the frame
            if random() < self.prob:
                continue
            combframes.append(frame)
        return combframes


class TimeWarpReverse(TimeWarp):
    """
    Applies a time-warping effect (reverse) to the audio.

    This effect uses the following method:
        - Split audio into frames, according to FPS value
        - Iterate over all frames
            - Sample a random value `x`
            - if `x < prob`:
                - Reverse the frame
    """

    def _timewarp(self, sliced_audio_frames: np.ndarray) -> list[np.ndarray]:
        combframes = []
        # Iterate over all the frames
        for frame in sliced_audio_frames:
            # If we trigger the effect, flip it horizontally
            if random() < self.prob:
                frame = np.flip(frame, axis=0)
            combframes.append(frame)
        return combframes


# Holds all augmentations that can be applied to Event objects
ALL_EVENT_AUGMENTATIONS = [
    LowpassFilter,
    HighpassFilter,
    MultibandEqualizer,
    Compressor,
    Chorus,
    Delay,
    Distortion,
    Phaser,
    Gain,
    GSMFullRateCompressor,
    MP3Compressor,
    PitchShift,
    SpeedUp,
    TimeWarpRemove,
    TimeWarpSilence,
    TimeWarpDuplicate,
    TimeWarpReverse,
    Preemphasis,
    Deemphasis,
    Fade,
    Clipping,
    Bitcrush,
    Limiter,
    HighShelfFilter,
    LowShelfFilter,
    Invert,
    Reverse,
]


# noinspection PyUnreachableCode
def validate_event_augmentation(augmentation_obj: Any) -> None:
    """
    Validates an augmentation class for the Event.

    In order to be valid, an augmentation must:
        - be callable;
        - be an instance of an augmentation class, not the class itself
        - inherit from the `audiblelight.augmentation.EventAugmentation` class;
        - define the `AUGMENTATION_TYPE` and `fx` properties;
        - have an `AUGMENTATION_TYPE` of "event" (not "scene", which applies to `audiblelight.core.Scene` objects).

    Arguments:
        augmentation_obj (Any): the augmentation object to validate

    Returns:
        None

    Raises:
        ValueError: if the augmentation object is invalid.
        AttributeError: if the augmentation object does not have a required property or attribute.
    """

    if not callable(augmentation_obj):
        raise ValueError("Augmentation object must be callable")

    if isinstance(augmentation_obj, type):
        raise ValueError(
            "Augmentation object must be an instance of a class, not the class itself"
        )

    if not issubclass(type(augmentation_obj), EventAugmentation):
        raise ValueError(
            "Augmentation object must be a subclass of `audiblelight.augmentation.EventAugmentation`"
        )

    for attr in ["fx", "AUGMENTATION_TYPE", "params"]:
        if not hasattr(augmentation_obj, attr):
            raise AttributeError(f"Augmentation object must have '{attr}' attribute")

    aug_type = getattr(augmentation_obj, "AUGMENTATION_TYPE", "")
    if aug_type != "event":
        raise ValueError(f"Augmentation type must be 'event', but got '{aug_type}'")
