#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate background noise for a Scene according to a given colour (white, pink...), value of β, or audio file.

The core functionality is adapted from [`colorednoise` by Felix Patzelt](https://github.com/felixpatzelt/colorednoise)
which is released under a permissive MIT license.
"""

import random
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import librosa
import numpy as np
from deepdiff import DeepDiff
from loguru import logger

from audiblelight import config, custom_types, utils

# This dictionary maps popular "names" to β values for generating noise
#  In general, higher β values cause more energy in high frequency parts of the power spectral density
NOISE_MAPPING = dict(pink=1, brown=2, red=2, blue=-1, white=0, violet=-2)


class Ambience:
    """
    Represents persistent background noise for a Scene.
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        channels: int,
        duration: custom_types.Numeric,
        alias: str,
        filepath: Optional[Union[str, Path]] = None,
        noise: Optional[Union[str, custom_types.Numeric]] = None,
        ref_db: Optional[custom_types.Numeric] = config.DEFAULT_REF_DB,
        sample_rate: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
        **kwargs,
    ):
        """
        Initialises persistent, invariant background noise for a Scene object.

        The audio used for ambience can be either a "colored" form of noise (e.g., white, blue, red, etc.), or a mono
        audio file. When an audio file is provided (by setting `filepath=...`), it will be tiled in both the horizontal
        and vertical directions to match the given number of channels and duration. Otherwise, the color of the noise
        must be specified (by setting `color=...`).

        Arguments:
            channels (int): the number of channels to use when generating ambience
            duration (Numeric): the duration (in seconds) for background ambience
            sample_rate (Numeric): the sample rate to use for generated ambience
            alias: Label to refer to this Ambience by inside the Scene
            filepath (str or Path): a path to an audio file on the disk. Must be provided when `noise` is None.
            noise (str): either the type of noise to generate, e.g. "white", "red", or an arbitrary numeric exponent to
                use when generating noise with `powerlaw_psd_gaussian`. Must be provided if `filepath` is None.
            ref_db (Numeric): the noise floor for the ambience
            kwargs: additional values passed to `powerlaw_psd_gaussian` when `noise` is not None.
        """

        # Basic attributes for the ambience, first three should be numeric
        self.channels = utils.sanitise_positive_number(channels, cast_to=int)
        self.sample_rate = utils.sanitise_positive_number(sample_rate, cast_to=int)
        self.duration = utils.sanitise_positive_number(duration)
        self.alias = alias

        # Parse the noise type: either an audio file, or a type of noise "color"
        if noise is None and filepath is not None:
            self.filepath, self.beta = utils.sanitise_filepath(filepath), None
        elif noise is not None and filepath is None:
            self.filepath, self.beta = None, _parse_beta(noise)
        elif noise is not None and filepath is not None:
            raise AttributeError(
                "Only one of `noise` or `filepath` should be provided."
            )
        else:
            raise AttributeError("One of `noise` or `filepath` must be provided")

        # Validate arguments passed to noise generation function and store them
        utils.validate_kwargs(powerlaw_psd_gaussian, **kwargs)
        self.noise_kwargs = kwargs

        # Validate noise floor
        #  should be a NEGATIVE number in dB, which we can test by inverting the sign and passing to our positive
        #  number validation function
        utils.sanitise_positive_number(-ref_db)
        self.ref_db = ref_db

        # Will be used to hold pre-rendered ambience
        self.audio = None

    def __eq__(self, other: Any) -> bool:
        """
        Compare two Ambience objects for equality.

        Returns:
            bool: True if two Ambience objects are equal, False otherwise
        """

        # Non-Ambience objects are always not equal
        if not isinstance(other, Ambience):
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

    def __str__(self) -> str:
        """
        Returns a string representation of the scene
        """
        loaded = "loaded" if self.is_audio_loaded else "unloaded"
        return f"'Ambience' with alias '{self.alias}' (currently {loaded})."

    def __repr__(self) -> str:
        """
        Returns representation of the scene as a JSON
        """
        return utils.repr_as_json(self)

    @property
    def is_audio_loaded(self) -> bool:
        """
        Returns True if noise is loaded and is valid (see `librosa.util.valid_audio` for more detail)
        """
        return self.audio is not None and librosa.util.valid_audio(self.audio)

    def load_ambience(
        self, ignore_cache: Optional[bool] = False, normalize: Optional[bool] = True
    ) -> np.ndarray:
        """
        Load the background ambience as an array with shape (channels, samples).
        """
        # If we've already loaded the audio, and it is still valid, we can return it straight away
        if self.is_audio_loaded and not ignore_cache:
            return self.audio

        total_samples = round(self.duration * self.sample_rate)

        # We want to use a "colored" form of noise
        if self.beta is not None:
            # This gives a matrix of shape (N_channels, N_samples)
            shape = (self.channels, total_samples)

            # Gaussian noise is a special case
            if self.beta == "gaussian":
                out = np.random.normal(
                    0,
                    1,
                    shape,
                )

            else:
                # It is normalized to approximately unit variance and zero mean
                out = powerlaw_psd_gaussian(self.beta, shape, **self.noise_kwargs)

        # Or, we want to use a noise file from disk
        else:
            ambient, _ = librosa.load(
                self.filepath, sr=self.sample_rate, mono=False, dtype=np.float32
            )
            ambient = utils.coerce2d(ambient)
            n_audio_channels, n_samples = ambient.shape

            # If the audio is multichannel, we need to check if we have the correct number of channels
            if n_audio_channels != self.channels:

                # Mono audio is always fine
                if n_audio_channels == 1:
                    ambient = ambient[0, :]

                # Otherwise, we have multichannel audio with an incorrect number of channels, so just take one randomly
                else:
                    logger.warning(
                        f"Passed audio has {n_audio_channels} channels, but expected {self.channels} channels. "
                        f"A random mono channel will be chosen from the audio."
                    )
                    ambient = ambient[random.choice(range(n_audio_channels)), :]

                # Audio at this point has shape (1, n_samples)
                #  We'll need to tile it along the channel dimension
                tile_channels = self.channels

            # Here, the audio is multichannel, but we have the correct number of channels
            #  So we can just use the audio directly, with repetitions along the time dimension
            #  We don't need to tile along the channel dimension as well
            else:
                tile_channels = 1

            repeats = -(-total_samples // n_samples)  # ceiling division
            # Tiles along both directions as required to get (n_channels, n_samples)
            out = np.tile(utils.coerce2d(ambient), (tile_channels, repeats))[
                :, :total_samples
            ]

        # Normalise noise to have max(abs(noise)) == 1 per channel
        if normalize:
            for c_idx in range(out.shape[0]):
                channel = out[c_idx]
                out[c_idx, :] = channel / np.max(np.abs(channel) + utils.tiny(channel))

        self.audio = out
        return self.audio

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        return dict(
            alias=self.alias,
            beta=self.beta,
            filepath=str(self.filepath) if self.filepath is not None else None,
            channels=self.channels,
            sample_rate=self.sample_rate,
            duration=self.duration,
            ref_db=self.ref_db,
            noise_kwargs=self.noise_kwargs,
        )

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        """
        Instantiate `Ambience` from a dictionary

        Arguments:
            input_dict (dict[str, Any]): the dictionary to instantiate from

        Returns:
            `Ambience` instance
        """

        # Sanitise inputs
        for k in [
            "alias",
            "filepath",
            "duration",
            "ref_db",
            "beta",
            "channels",
        ]:
            if k not in input_dict:
                raise KeyError(f"Missing key: '{k}'")

        return cls(
            channels=input_dict["channels"],
            sample_rate=input_dict["sample_rate"],
            alias=input_dict["alias"],
            filepath=input_dict["filepath"],
            duration=input_dict["duration"],
            noise=input_dict["beta"],
            ref_db=input_dict["ref_db"],
            **input_dict["noise_kwargs"],
        )


# noinspection PyUnreachableCode
def powerlaw_psd_gaussian(
    beta: custom_types.Numeric,
    shape: Union[int, Iterable[int]],
    fmin: Optional[custom_types.Numeric] = 0.0,
    seed: Optional[int] = utils.SEED,
) -> np.ndarray:
    """Generate Gaussian (1 / f) ** β noise.

    Based on: Timmer, J. and Koenig, M.: On generating power law noise. Astron. Astrophys. 300, 707-710 (1995)

    Arguments:
        beta (float): The power-spectrum of the generated noise is proportional to S(f) = (1 / f) ** β,
        shape (int or iterable): The output has the given shape, and the desired power spectrum in the last coordinate.
            That is, the last dimension is taken as time, and all other components are independent.
        fmin (float): Low-frequency cutoff. Default: 0 corresponds to original paper. The largest possible
            value is fmin = 0.5, the Nyquist frequency. The output for this value is white noise.
        seed (int): Seed to use when creating the normal distribution.

    Returns:
        np.ndarray: the noise samples in the shape (channels, samples)

    Examples:
        # Generate monophonic pink noise with 5 samples
        >>> noise = powerlaw_psd_gaussian(1, 5)
        >>> noise.shape
        (5,)

        # Generate quadraphonic pink noise with 10 samples
        >>> noise = powerlaw_psd_gaussian(1, (4, 10))
        >>> noise.shape
        (4, 10)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(shape, (np.integer, int)):
        size = [shape]
    elif isinstance(shape, Iterable):
        size = list(shape)
    else:
        raise ValueError(
            f"Argument `shape` must be of type int or Iterable[int] but got {type(shape)}"
        )

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we assume a sample rate of one)
    #  Use fft functions for real output (-> hermitian spectrum)
    f = np.fft.rfftfreq(samples)  # type: ignore # mypy 1.5.1 has problems here

    # Validate / normalise fmin
    fmin = utils.sanitise_positive_number(fmin)
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / (samples + utils.tiny(samples)))  # Low frequency cutoff
    else:
        raise ValueError(
            f"Argument `fmin` must be chosen between 0 and 0.5 but got {fmin:.2f}."
        )

    # Build scaling factors for all frequencies
    s_scale = f
    ix = np.sum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-beta / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / (samples + utils.tiny(samples))

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    #  dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(np.newaxis,) * dims_to_add + (Ellipsis,)]

    # Prepare random number generator
    random_state = np.random.default_rng(seed)
    normal_dist = random_state.normal

    # Generate scaled random power + phase
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=samples, axis=-1)
    y /= sigma

    return y


def _parse_beta(noise: Any) -> Union[float, str]:
    """
    Parses the noise exponential term from either a string representation of a color (white) or a number.
    """
    # String color must be in the dictionary
    if isinstance(noise, str):
        if noise in NOISE_MAPPING.keys():
            return NOISE_MAPPING[noise]
        elif noise.lower() == "gaussian":
            return "gaussian"
        else:
            keys = ", ".join(k for k in NOISE_MAPPING.keys())
            raise KeyError(f"Expected a string in {keys} but got {noise}.")

    # Otherwise, exponent must be numeric
    elif isinstance(noise, custom_types.Numeric):
        return noise

    # Must provide either a color or exponent
    else:
        raise TypeError(
            f"Expected either a string or numeric input, but got {type(noise)}."
        )
