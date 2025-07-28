#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate background noise for a Scene according to a given colour (white, pink...) or value of β.

The core functionality is adapted from [`colorednoise` by Felix Patzelt](https://github.com/felixpatzelt/colorednoise)
which is released under a permissive MIT license.
"""

from typing import Any, Union, Iterable, Optional

import numpy as np
import librosa

from audiblelight import utils


# This dictionary maps popular "names" to β values for generating noise
#  In general, higher β values cause more energy in high frequency parts of the power spectral density
NOISE_MAPPING = dict(
    pink=1,
    brown=2,
    red=2,
    blue=-1,
    white=0,
    violet=-2
)


class Ambience:
    """
    Represents persistent background noise for a Scene.
    """

    # noinspection PyTypeChecker
    def __init__(
            self,
            shape: tuple[int, int],
            color: Optional[str] = None,
            exponent: Optional[utils.Numeric] = None,
            ref_db: Optional[utils.Numeric] = utils.REF_DB,
            **kwargs
    ):
        """
        Initialises persistent, invariant background noise for a Scene object.

        Currently, only "colored" forms of noise (white, blue, red, etc.) are supported, with an arbitrary channel count.

        Arguments:
            shape (tuple): the shape of generated noise, in the form (channels, samples)
            color (str): the type of noise to generate, e.g. "white", "red", must be provided if `exponent` is None
            exponent (Numeric): the coefficient for the generated noise, must be provided if `color` is None
            ref_db (Numeric): the noise floor for the ambience
            kwargs: additional values passed to `powerlaw_psd_gaussian`.
        """

        # Parse the exponent for the noise generation
        #  This can either be a color (e.g., "pink", "white") or a numeric value
        self.beta = _parse_beta(color, exponent)

        # Validate shape for the noise
        if len(shape) != 2:
            raise ValueError(f"Expected `shape` in the form (n_channels, n_samples), but got {shape} instead!")
        self.shape = shape

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

    @property
    def is_audio_loaded(self) -> bool:
        """
        Returns True if noise is loaded and is valid (see `librosa.util.valid_audio` for more detail)
        """
        return self.audio is not None and librosa.util.valid_audio(self.audio)

    def load_ambience(self, ignore_cache: bool = False) -> np.ndarray:
        """
        Load the background ambience as an array with shape (channels, samples).
        """
        # If we've already loaded the audio, and it is still valid, we can return it straight away
        if self.is_audio_loaded and not ignore_cache:
            out = self.audio

        # Otherwise, we need to create the ambience from scratch
        else:
            # This gives a matrix of shape (N_channels, N_samples)
            #  It is normalized to unit variance and has a zero mean (SD ~
            out = powerlaw_psd_gaussian(self.beta, self.shape, **self.noise_kwargs)
            # Now we scale to match the desired noise floor
            #  This is taken from SpatialScaper
            # TODO: second arg here should be computed with RMS, but this leads to clipping
            scaler = utils.db_to_multiplier(self.ref_db, np.mean(np.abs(out)))
            out *= scaler

        # Set the audio to our property and return
        self.audio = out
        return self.audio

    def to_dict(self) -> dict:
        """
        Returns metadata for this object as a dictionary
        """
        return dict(
            beta=self.beta,
            shape=self.shape,
            ref_db=self.ref_db,
            **self.noise_kwargs
        )


def powerlaw_psd_gaussian(
        beta: utils.Numeric,
        shape: Union[int, Iterable[int]],
        fmin: Optional[utils.Numeric] = 0.0,
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
        raise ValueError(f"Argument `shape` must be of type int or Iterable[int] but got {type(shape)}")

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we assume a sample rate of one)
    #  Use fft functions for real output (-> hermitian spectrum)
    f = np.fft.rfftfreq(samples)  # type: ignore # mypy 1.5.1 has problems here

    # Validate / normalise fmin
    fmin = utils.sanitise_positive_number(fmin)
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / samples)  # Low frequency cutoff
    else:
        raise ValueError(f"Argument `fmin` must be chosen between 0 and 0.5 but got {fmin:.2f}.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = np.sum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-beta / 2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.  # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w ** 2)) / samples

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
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=samples, axis=-1)
    y /= sigma

    return y


def _parse_beta(color: Any, exponent: Any) -> float:
    """
    Parses the noise exponential term from either a string representation of a color (white) or a number.
    """
    # Both values are provided
    if color is not None and exponent is not None:
        # The provided color does not have the same exponent as the provided value
        if color in NOISE_MAPPING.keys() and NOISE_MAPPING[color] != exponent:
            raise ValueError("Both `color` and `exponent` were provided, however the values do not match: "
                             f"expected {NOISE_MAPPING[color]}, but got {exponent}.")
        # else: use "color" not "exponent", but it doesn't really matter as both would give the same results

    # String color must be in the dictionary
    if color is not None:
        if color in NOISE_MAPPING.keys():
            return NOISE_MAPPING[color]
        elif not isinstance(color, str):
            raise TypeError(f"`color` must be a string but got {type(color)}")
        else:
            keys = ", ".join(k for k in NOISE_MAPPING.keys())
            raise KeyError(f"`color` must be a string in {keys} but got {color}.")

    # Otherwise, exponent must be numeric
    elif exponent is not None:
        if isinstance(exponent, utils.Numeric):
            return exponent
        else:
            raise TypeError(f"`exponent` must be a numeric value, but got {type(exponent)}")

    # Must provide either a color or exponent
    else:
        raise TypeError("Either one of `color` or `exponent` must be provided.")
