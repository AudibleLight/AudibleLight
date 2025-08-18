#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides classes and functions for handling spatial and non-spatial audio augmentations.

Non-spatial augmentations
-------------------------

Many of these classes are wrappers around effects from [`pedalboard`](https://spotify.github.io/pedalboard/). The
wrappers enable parameters for each FX to be randomly sampled from acceptable default distributions, or a user
provided distribution. Some of the FX are not implemented in `pedalboard` (e.g., `Equalizer`, which simulates a
multi-band, parametric equalizer). The `TimeWarpXXXX` effects are taken from
[this paper](https://arxiv.org/pdf/2502.06364).

"""

from typing import Any, Callable, Iterator, Optional, Type, Union

import numpy as np
from deepdiff import DeepDiff
from pedalboard import time_stretch
from scipy import stats

from audiblelight import utils
from audiblelight.synthesize import pad_or_truncate_audio
from audiblelight.utils import DistributionLike

BUFFER_SIZE = 8192
MIN_FPS, MAX_FPS = 0.5, 5


def _identity(input_array: np.ndarray, *_, **__) -> np.ndarray:
    return input_array


class Augmentation:
    """
    Base class for all augmentation objects to inherit from.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        kwargs: any additional keyword arguments to pass to the effect.

    Properties:
        fx (Callable): the callable function applied to the audio. Can be from Pedalboard, Librosa, PyRubberband...
        params (dict): the arguments passed to `fx`. Will be serialised inside `to_json`.

    """

    def __init__(
        self,
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        **kwargs,
    ):
        self.sample_rate = int(utils.sanitise_positive_number(sample_rate))
        self.buffer_size = int(utils.sanitise_positive_number(buffer_size))
        self.reset = reset
        self.fx: Union[Callable, list[Callable]] = _identity
        self.params = dict()

    @staticmethod
    def sample_value(
        override: Optional[Union[utils.Numeric, utils.DistributionLike]],
        default_dist: utils.DistributionLike,
    ) -> utils.Numeric:
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
        elif isinstance(override, utils.Numeric):
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
            out = fx(out, self.sample_rate, self.buffer_size, self.reset)

        # Temporary convert mono to stereo for pad function
        if out.ndim == 1:
            out = np.expand_dims(out, 0)

        # Pad or truncate the audio to keep the same dims
        trunc = pad_or_truncate_audio(out, max(input_array.shape))

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
            buffer_size=self.buffer_size,
            reset=self.reset,
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


class LowpassFilter(Augmentation):
    """
    Applies a low-pass filter to the audio.

    By default, the cutoff frequency for the filter will be sampled randomly between 5512 and 22050 Hz. Either the
    exact cutoff frequency or a distribution to sample this from can be provided as arguments to the function.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        cutoff_frequency_hz (Union[utils.Numeric, utils.DistributionLike]): the cutoff frequency for the filter, or a
            distribution-like object to sample this from. Will default to sampling from uniform distribution
            between 5512 and 22050 Hz if not provided.
    """

    MIN_FREQ, MAX_FREQ = 5512, 22050

    def __init__(
        self,
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        cutoff_frequency_hz: Optional[
            Union[utils.Numeric, utils.DistributionLike]
        ] = None,
    ):
        # Initialise the parent class
        super().__init__(sample_rate, buffer_size, reset)

        # Handle sampling the cutoff frequency
        #  Can be a numeric value or a distribution passed by the user
        #  Sampled from this default distribution if not given
        self.cutoff_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                cutoff_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )

        # Initialise the FX with the required parameter
        from pedalboard import LowpassFilter as PBLowpassFilter

        self.fx = PBLowpassFilter(cutoff_frequency_hz=self.cutoff_frequency_hz)

        self.params = dict(cutoff_frequency_hz=self.cutoff_frequency_hz)


class HighpassFilter(Augmentation):
    """
    Applies a high-pass filter to the audio.

    By default, the cutoff frequency for the filter will be sampled randomly between 32 and 1024 Hz. Either the
    exact cutoff frequency or a distribution to sample this from can be provided as arguments to the function.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        cutoff_frequency_hz (Union[utils.Numeric, utils.DistribtionLike]): the cutoff frequency for the filter, or a
            distribution-like object to sample this from. Will default to sampling from uniform distribution
            between 32 and 1024 Hz if not provided.
    """

    MIN_FREQ, MAX_FREQ = 32, 1024

    def __init__(
        self,
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        cutoff_frequency_hz: Optional[
            Union[utils.Numeric, utils.DistributionLike]
        ] = None,
    ):
        # Initialise the parent class
        super().__init__(sample_rate, buffer_size, reset)

        # Handle sampling the cutoff frequency
        #  Can be a numeric value or a distribution passed by the user
        #  Sampled from this default distribution if not given
        self.cutoff_frequency_hz = utils.sanitise_positive_number(
            self.sample_value(
                cutoff_frequency_hz,
                stats.uniform(self.MIN_FREQ, self.MAX_FREQ - self.MIN_FREQ),
            )
        )

        # Initialise the FX with the required parameter
        from pedalboard import HighpassFilter as PBHighpassFilter

        self.fx = PBHighpassFilter(cutoff_frequency_hz=self.cutoff_frequency_hz)

        self.params = dict(cutoff_frequency_hz=self.cutoff_frequency_hz)


class Equalizer(Augmentation):
    """
    Applies equalization to the audio.

    The Equalizer applies N peak filter objects to the audio. The gain, frequency, and Q of each filter can be
    set independently, or randomised from within an acceptable range. By default, between 1 and 8 individual peak
    filters are applied, with randomly selected gain, frequency, and Q of each filter.

    Using this class, it is possible to create a multiband equalizer similar to the "parametric EQ" plugins often
    featured in digital audio workstations. For additional flexibility, consider combining with both `HighpassFilter`
    and `LowpassFilter`.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
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
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        n_bands: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        gain_db: Optional[
            Union[list[utils.Numeric], utils.Numeric, utils.DistributionLike]
        ] = None,
        cutoff_frequency_hz: Optional[
            Union[list[utils.Numeric], utils.Numeric, utils.DistributionLike]
        ] = None,
        q: Optional[
            Union[list[utils.Numeric], utils.Numeric, utils.DistributionLike]
        ] = None,
    ):
        super().__init__(sample_rate, buffer_size, reset)

        # The number of frequency bands we'll be applying
        self.n_bands = int(
            utils.sanitise_positive_number(
                self.sample_value(
                    n_bands,
                    stats.uniform(self.MIN_BANDS, self.MAX_BANDS - self.MIN_BANDS),
                )
            )
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

        # Given the parameter settings, create the filters
        self.fx = self.create_filters()
        self.params = dict(
            n_bands=int(self.n_bands),
            gain_db=self.gain_db,
            cutoff_frequency_hz=self.cutoff_frequency_hz,
            q=self.q,
        )

    # noinspection PyUnreachableCode,PyUnresolvedReferences
    def sample_peak_filter_params(
        self,
        override: Union[utils.Numeric, list[utils.Numeric], utils.DistributionLike],
        default_dist: DistributionLike,
    ) -> list[utils.Numeric]:
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
        elif isinstance(override, utils.Numeric):
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


class Compressor(Augmentation):
    """
    Applies compression to the audio signal.

    A dynamic range compressor, used to reduce the volume of loud sounds and "compress" the loudness of the signal. For
    a lossy compression algorithm that introduces noise or artifacts, see `MP3Compressor` or `GSMCompressor`.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
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
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        threshold_db: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        ratio: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        attack_ms: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        release_ms: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import Compressor as PBCompressor

        super().__init__(sample_rate, buffer_size, reset)

        # Set all FX parameters
        self.threshold_db = int(
            (
                self.sample_value(
                    threshold_db,
                    stats.uniform(self.MIN_THRESHOLD_DB, abs(self.MAX_THRESHOLD_DB)),
                )
            )
        )
        if self.threshold_db > 0:
            self.threshold_db = -self.threshold_db

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

        self.fx = PBCompressor(
            self.threshold_db, self.ratio, self.attack_ms, self.release_ms
        )
        self.params = dict(
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
        )


class Chorus(Augmentation):
    """
    Applies chorus to the audio.

    This audio effect can be controlled via the speed and depth of the LFO controlling the frequency response,
    a mix control, a feedback control, and the centre delay of the modulation.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
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
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        rate_hz: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        depth: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        centre_delay_ms: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        feedback: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        mix: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import Chorus as PBChorus

        super().__init__(sample_rate, buffer_size, reset)

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

        self.fx = PBChorus(
            self.rate_hz, self.depth, self.centre_delay_ms, self.feedback, self.mix
        )
        self.params = dict(
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_delay_ms=self.centre_delay_ms,
            feedback=self.feedback,
            mix=self.mix,
        )


class Distortion(Augmentation):
    """
    Applies distortion to the audio.

    Applies a non-linear (tanh, or hyperbolic tangent) waveshaping function to apply harmonically pleasing distortion
    to a signal.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        drive_db: the dB level of the distortion effect. By default, will be sampled between 10 and 30 dB.
    """

    MIN_DRIVE, MAX_DRIVE = 10, 30

    def __init__(
        self,
        sample_rate: utils.Numeric = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        drive_db: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import Distortion as PBDistortion

        super().__init__(sample_rate, buffer_size, reset)
        self.drive_db = utils.sanitise_positive_number(
            self.sample_value(
                drive_db, stats.uniform(self.MIN_DRIVE, self.MAX_DRIVE - self.MIN_DRIVE)
            )
        )
        self.fx = PBDistortion(drive_db=self.drive_db)
        self.params = dict(drive_db=self.drive_db)


class Phaser(Augmentation):
    """
    Applies a phaser to the audio.

    A 6 stage phaser that modulates first order all-pass filters to create sweeping notches in the magnitude frequency
    response. This audio effect can be controlled with standard phaser parameters: the speed and depth of the LFO
    controlling the frequency response, a mix control, a feedback control, and the centre frequency of the modulation.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
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
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        rate_hz: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        depth: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        centre_frequency_hz: Optional[
            Union[utils.Numeric, utils.DistributionLike]
        ] = None,
        feedback: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        mix: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import Phaser as PBPhaser

        super().__init__(sample_rate, buffer_size, reset)
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

        self.fx = PBPhaser(
            self.rate_hz, self.depth, self.centre_frequency_hz, self.feedback, self.mix
        )
        self.params = dict(
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_frequency_hz=self.centre_frequency_hz,
            feedback=self.feedback,
            mix=self.mix,
        )


class Delay(Augmentation):
    """
    Applies delay to the audio.

    A digital delay plugin with controllable delay time, feedback percentage, and dry/wet mix.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        delay: the delay time for the effect, in seconds. By default, sampled between 0.01 and 1.0 seconds.
        feedback: the feedback of the effect. By default, sampled between 0.0 and 0.9.
        mix: the dry/wet mix of the effect. By default, sampled between 0.1 and 0.5
    """

    MIN_DELAY, MAX_DELAY = 0.01, 1.0
    MIN_FEEDBACK, MAX_FEEDBACK = 0.1, 0.9
    MIN_MIX, MAX_MIX = 0.1, 0.5

    def __init__(
        self,
        sample_rate: utils.Numeric = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        delay: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        feedback: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
        mix: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import Delay as PBDelay

        super().__init__(sample_rate, buffer_size, reset)
        self.delay = utils.sanitise_positive_number(
            self.sample_value(
                delay, stats.uniform(self.MIN_DELAY, self.MAX_DELAY - self.MIN_DELAY)
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

        self.fx = PBDelay(self.delay, self.feedback, self.mix)
        self.params = dict(
            delay=self.delay,
            feedback=self.feedback,
            mix=self.mix,
        )


class Gain(Augmentation):
    """
    Applies gain (volume) to the audio.

    A gain plugin that increases or decreases the volume of a signal by amplifying or attenuating it by the
    provided value (in decibels). No distortion or other effects are applied.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        gain_db: the gain to apply to the signal. By default, sampled between -10 and 10 dB.
    """

    MIN_GAIN, MAX_GAIN = -10, 10

    def __init__(
        self,
        sample_rate: utils.Numeric = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        gain_db: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import Gain as PBGain

        super().__init__(sample_rate, buffer_size, reset)
        self.gain_db = self.sample_value(
            gain_db, stats.uniform(self.MIN_GAIN, self.MAX_GAIN - self.MIN_GAIN)
        )
        self.fx = PBGain(gain_db=self.gain_db)
        self.params = dict(gain_db=self.gain_db)


class GSMFullRateCompressor(Augmentation):
    """
    Applies GSM compression to the audio.

    An audio degradation/compression plugin that applies the GSM “Full Rate” compression algorithm to emulate the
    sound of a 2G cellular phone connection. This plugin internally resamples the input audio to a fixed sample rate
    of 8kHz (required by the GSM Full Rate codec), although the quality of the resampling algorithm can be specified.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        quality: the quality of the resampling. By default, will be sampled between 0 and 3 (inclusive).
    """

    # Don't use the highest resampling quality (4) as it is much slower than the others
    QUALITIES = range(4)

    def __init__(
        self,
        sample_rate: utils.Numeric = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        quality: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import GSMFullRateCompressor as PBGSMFullRateCompressor
        from pedalboard import Resample

        super().__init__(sample_rate, buffer_size, reset)
        self.quality = int(
            utils.sanitise_positive_number(
                self.sample_value(quality, lambda: np.random.choice(self.QUALITIES))
            )
        )
        self.fx = PBGSMFullRateCompressor(quality=Resample.Quality(self.quality))
        self.params = dict(quality=self.quality)


class MP3Compressor(Augmentation):
    """
    Applies the LAME MP3 encoder in real-time to add compression artifacts to the audio stream.

    Currently only supports variable bit-rate mode (VBR) and accepts a floating-point VBR quality value
    (between 0.0 and 10.0; lower is better). Note that the MP3 format only supports 8kHz, 11025Hz, 12kHz, 16kHz,
    22050Hz, 24kHz, 32kHz, 44.1kHz, and 48kHz audio; if an unsupported sample rate is provided, an exception will be
    thrown at processing time.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): the size of the buffer for the audio effect.
        reset (bool): if True, the internal state of the FX will be reset every time it is called.
        vbr_quality: the quality of the resampling. By default, will be sampled between 2 and 10.
    """

    VBR_MIN, VBR_MAX = 2.001, 9.999

    def __init__(
        self,
        sample_rate: utils.Numeric = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        vbr_quality: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        from pedalboard import MP3Compressor as PBMP3Compressor

        super().__init__(sample_rate, buffer_size, reset)
        self.vbr_quality = utils.sanitise_positive_number(
            self.sample_value(
                vbr_quality, stats.uniform(self.VBR_MIN, self.VBR_MAX - self.VBR_MIN)
            )
        )
        self.fx = PBMP3Compressor(vbr_quality=self.vbr_quality)
        self.params = dict(vbr_quality=self.vbr_quality)


class PitchShift(Augmentation):
    """
    Applies pitch-shifting to the audio.

    Internally, this function uses the `time_shift` function in pedalboard, specifying the `pitch_shift` as an array
    with the same dims as `input_audio`. From our initial testing, we found this algorithm to be significantly faster
    than many other existing pitch shifting algorithms, including `librosa` and `pyrubberband`.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): ignored for this class.
        reset (bool): ignored for this class.
        semitones: the number of semitones to shift the audio by. By default, will be sampled from between +/- 3
            semitones (i.e., up or down a minor third).
    """

    MIN_SEMITONES, MAX_SEMITONES = -3, 3

    def __init__(
        self,
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        semitones: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        super().__init__(sample_rate, buffer_size, reset)

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

    def process(self, input_array: np.ndarray, *args) -> np.ndarray:
        """
        Apply the effect to the input audio.
        """
        # Simply return the input if we're not using pitch shifting
        if self.semitones == 0:
            return input_array
        return super().process(input_array)


class TimeShift(Augmentation):
    """
    Applies time-stretching to the audio.

    Using a higher stretch_factor will shorten the audio - i.e., a stretch_factor of 2.0 will double the speed of the
    audio and halve the length of the audio, without changing the pitch of the audio. When the output audio is shorter
    than the input, it will be right-padded with zeros to maintain the correct dim. When the output audio is longer
    than the input, it will be truncated to maintain the correct dim.

    Arguments:
        sample_rate (utils.Numeric): the sample rate for the effect to use.
        buffer_size (utils.Numeric): ignored for this class.
        reset (bool): ignored for this class.
        stretch_factor: the time-stretching factor to apply. Values above 1 will increase the speed of the audio, while
            values below 1 will decrease the speed. A value of 1 will have no effect. By default, will be sampled
            from between 0.7 and 1.5.
    """

    MIN_SHIFT, MAX_SHIFT = 0.7, 1.5

    def __init__(
        self,
        sample_rate: Optional[utils.Numeric] = utils.SAMPLE_RATE,
        buffer_size: Optional[utils.Numeric] = BUFFER_SIZE,
        reset: Optional[bool] = True,
        stretch_factor: Optional[Union[utils.Numeric, utils.DistributionLike]] = None,
    ):
        super().__init__(sample_rate, buffer_size, reset)
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

    def process(self, input_array: np.ndarray, *args) -> np.ndarray:
        """
        Apply the effect to the input audio.
        """
        # Identity operation
        if self.stretch_factor == 1.0:
            return input_array
        return super().process(input_array)


# class TimeWarpSilence(Augmentation):
#     PROB = 0.1
#
#     def __init__(self, sample_rate: utils.Numeric = utils.SAMPLE_RATE, **kwargs):
#         super().__init__(sample_rate)
#         self.fps = kwargs.get('fps', np.random.uniform(MIN_FPS, MAX_FPS))
#         self.seed = kwargs.get('seed', utils.SEED)
#         self.prob = kwargs.get("prob", self.PROB)
#
#         self.params = dict(
#             fps=float(self.fps),
#             seed=int(self.seed),
#             prob=self.prob
#         )
#
#     def process(
#             self,
#             input_array: np.ndarray,
#             *args
#     ) -> np.ndarray:
#         """
#         Apply the effect to the input audio.
#         """
#         # Set the random seed according to the value defined in `__init__`, for reproducibility
#         np.random.seed(self.seed)
#
#         # Slice the audio into frames
#         sliced = slice_frames(input_array, args[0], self.fps)
#         combframes = []
#
#         # Iterate over all the frames
#         for frame in sliced:
#             # If we trigger the effect, zero the frame
#             if np.random.uniform(0., 1.) < self.prob:
#                 frame = np.zeros(len(frame))
#             combframes.append(frame)
#
#         # If we've never triggered the effect, return the original audio
#         try:
#             transformed = np.concatenate(combframes)
#         except ValueError:
#             return input_array
#         else:
#             return pad_or_truncate_audio(transformed, len(input_array))
#
#
# class TimeWarpDuplicate(Augmentation):
#     PROB = 0.1
#
#     def __init__(self, sample_rate: utils.Numeric = utils.SAMPLE_RATE, **kwargs):
#         super().__init__(sample_rate)
#         self.fps = kwargs.get('fps', np.random.uniform(MIN_FPS, MAX_FPS))
#         self.seed = kwargs.get('seed', utils.SEED)
#         self.prob = kwargs.get("prob", self.PROB)
#
#         self.params = dict(
#             fps=float(self.fps),
#             seed=int(self.seed),
#             prob=self.prob
#         )
#
#     def process(
#             self,
#             input_array: np.ndarray,
#             *args
#     ) -> np.ndarray:
#         """
#         Apply the effect to the input audio.
#         """
#         # Set the random seed according to the value defined in `__init__`, for reproducibility
#         np.random.seed(self.seed)
#
#         # Slice the audio into frimes
#         sliced = slice_frames(input_array, args[0], self.fps)
#         combframes = []
#
#         # Iterate over all the frames
#         for frame in sliced:
#             # If we trigger the effect, append the frame to the list twice
#             if np.random.uniform(0., 1.) < self.prob:
#                 combframes.append(frame)
#             combframes.append(frame)
#
#         # If we've never triggered the effect, just return the input audio
#         try:
#             transformed = np.concatenate(combframes)
#         except ValueError:
#             return input_array
#         else:
#             return pad_or_truncate_audio(transformed, len(input_array))
#
#
# class TimeWarpRemove(Augmentation):
#     PROB = 0.1
#
#     def __init__(self, sample_rate: utils.Numeric = utils.SAMPLE_RATE, **kwargs):
#         super().__init__(sample_rate)
#         self.fps = kwargs.get('fps', np.random.uniform(MIN_FPS, MAX_FPS))
#         self.seed = kwargs.get('seed', utils.SEED)
#         self.prob = kwargs.get("prob", self.PROB)
#
#         self.params = dict(
#             fps=float(self.fps),
#             seed=int(self.seed),
#             prob=self.prob
#         )
#
#     def process(
#             self,
#             input_array: np.ndarray,
#             *args
#     ) -> np.ndarray:
#         """
#         Apply the effect to the input audio.
#         """
#         # Set the random seed according to the value defined in `__init__`, for reproducibility
#         np.random.seed(self.seed)
#
#         # Slice the audio into frames
#         sliced = slice_frames(input_array, args[0], self.fps)
#         combframes = []
#
#         # Iterate over all the frames
#         for frame in sliced:
#             # If we trigger the effect, skip the frame
#             if np.random.uniform(0., 1.) < self.prob:
#                 continue
#             combframes.append(frame)
#
#         # If we've never triggered the effect, just return the input audio
#         try:
#             transformed = np.concatenate(combframes)
#         except ValueError:
#             return input_array
#         else:
#             return pad_or_truncate_audio(transformed, len(input_array))
#
#
# class TimeWarpReverse(Augmentation):
#     PROB = 0.1
#
#     def __init__(self, sample_rate: utils.Numeric = utils.SAMPLE_RATE, **kwargs):
#         super().__init__(sample_rate)
#         self.fps = kwargs.get('fps', np.random.uniform(MIN_FPS, MAX_FPS))
#         self.seed = kwargs.get('seed', utils.SEED)
#         self.prob = kwargs.get("prob", self.PROB)
#
#         self.params = dict(
#             fps=float(self.fps),
#             seed=int(self.seed),
#             prob=self.prob
#         )
#
#     def process(
#             self,
#             input_array: np.ndarray,
#             *args
#     ) -> np.ndarray:
#         """
#         Apply the effect to the input audio.
#         """
#         # Set the random seed according to the value defined in `__init__`, for reproducibility
#         np.random.seed(self.seed)
#
#         # Slice the audio according to the number of FPS
#         sliced = slice_frames(input_array, args[0], self.fps)
#         combframes = []
#
#         # Iterate over all the frames
#         for frame in sliced:
#             # If we trigger the effect, flip the frame horizontally
#             if np.random.uniform(0., 1.) < self.prob:
#                 frame = np.flip(frame, axis=0)
#             combframes.append(frame)
#
#         # Combine all the frames back into a single array
#         try:
#             transformed = np.concatenate(combframes)
#         # If we've never triggered the effect, just return the input audio
#         except ValueError:
#             return input_array
#         else:
#             return pad_or_truncate_audio(transformed, len(input_array))
#
#
# class Remix(Augmentation):
#     pass
#
#
# def slice_frames(inp_audio: np.ndarray, sample_rate: float, fps: float) -> np.ndarray:
#     """
#     Slice audio into non-overlapping frames
#     """
#     # This code is just taken from `librosa.util.frame`
#     axis = 0
#     frame_length = int(sample_rate / fps)
#     hop_length = frame_length
#     x = np.array(inp_audio, copy=False, subok=False)
#
#     # put our new within-frame axis at the end for now
#     out_strides = x.strides + tuple([x.strides[axis]])
#
#     # Reduce the shape on the framing axis
#     x_shape_trimmed = list(x.shape)
#     x_shape_trimmed[axis] -= frame_length - 1
#     out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
#     xw = np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape, subok=False, writeable=False)
#     xw = np.moveaxis(xw, -1, axis + 1)
#
#     # Downsample along the target axis
#     slices = [slice(None)] * xw.ndim
#     slices[axis] = slice(0, None, hop_length)
#     return xw[tuple(slices)]
