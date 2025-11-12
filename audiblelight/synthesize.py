#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules and functions for "synthesizing" outputs (including audio, video, image, metadata)"""

from collections import Counter
from time import time
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from loguru import logger
from scipy import fft, signal

from audiblelight import config, custom_types, utils
from audiblelight.ambience import Ambience
from audiblelight.core import Scene
from audiblelight.event import Event

# See https://dcase.community/challenge2024/task-audio-and-audiovisual-sound-event-localization-and-detection-with-source-distance-estimation
DCASE_2024_COLUMNS = [
    "frame_number",
    "active_class_index",
    "source_number_index",
    "azimuth",
    "elevation",
    "distance",
]


def apply_snr(x: np.ndarray, snr: custom_types.Numeric) -> np.ndarray:
    """
    Scale an audio signal to a given maximum SNR.

    Taken from [`SpatialScaper`](https://github.com/marl/SpatialScaper/blob/main/spatialscaper/spatialize.py#L147)

    Return:
        np.ndarray: the scaled audio signal
    """
    return x * snr / np.abs(x).max(initial=1e-15)


def db_to_multiplier(db: custom_types.Numeric, x: custom_types.Numeric) -> float:
    """
    Calculates the multiplier factor from a decibel (dB) value that, when applied to x, adjusts its amplitude to
    reflect the specified dB. The relationship is based on the formula 20 * log10(factor * x) ≈ db.

    Adapted from [`SpatialScaper`](https://github.com/marl/SpatialScaper/blob/main/spatialscaper/utils.py#L287)

    Arguments:
        db (float): The target decibel change to be applied.
        x  (float): The original amplitude of x

    Returns:
        float: The multiplier factor.
    """
    # Need to add a small value here to prevent divide by zero errors
    #  These can cause the audio buffer to become filled with NaNs, which leads to errors later on
    return 10 ** (db / 20) / (x + utils.tiny(x))


def time_invariant_convolution(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """
    Convolve a static, time-invariant impulse response with an audio input.

    Arguments:
        audio (np.ndarray): Input audio waveform with shape (n_samples,)
        ir (np.ndarray): Input impulse response waveform with shape (n_samples, n_ir_channels)

    Returns:
        np.ndarray: The convolved audio signal with shape (n_samples, n_ir_channels)

    Raises:
        ValueError: When input arrays do not match expected shapes
    """
    # Sanitise inputs
    if audio.ndim != 1:
        raise ValueError(
            f"Only mono input is supported, but got {audio.ndim} dimensions!"
        )
    if ir.ndim != 2:
        raise ValueError(
            f"Expected shape of IR should be (n_samples, n_channels), but got ({ir.shape}) instead"
        )

    # Shape of both arrays should be (n_samples, n_channels)
    #  where audio[n_channels] == 1, ir[n_channels] >= 1,
    #  and audio[n_samples] may not equal ir[n_samples]
    audio = np.expand_dims(audio, 1)

    # Perform the convolution using FFT method
    #  the output shape will be ((n_ir_samples + n_audio_samples) - 1, n_ir_channels)
    #  we will truncate to match the expected number of samples later
    convolve: np.ndarray = signal.fftconvolve(audio, ir, mode="full", axes=0)

    # Transpose so we get (channels, samples)
    return convolve.T


def stft(
    y: np.ndarray,
    fft_size: Optional[custom_types.Numeric] = config.FFT_SIZE,
    win_size: Optional[custom_types.Numeric] = config.WIN_SIZE,
    hop_size: Optional[custom_types.Numeric] = config.HOP_SIZE,
    stft_dims_first: Optional[bool] = True,
) -> np.ndarray:
    """
    Compute the STFT for a given input signal
    """
    # Generate the window function
    window = np.sin(np.pi / win_size * np.arange(win_size)) ** 2

    # Compute padding and pad the input signal
    n_frames = 2 * int(np.ceil(y.shape[-1] / (2.0 * hop_size))) + 1
    pad_width = [(0, 0)] * (y.ndim - 1) + [
        (win_size - hop_size, n_frames * hop_size - y.shape[-1])
    ]
    y_padded = np.pad(y, pad_width, mode="constant")

    # Use stride tricks to efficiently extract windows
    shape = y_padded.shape[:-1] + (win_size, n_frames)
    strides = y_padded.strides[:-1] + (
        y_padded.strides[-1],
        y_padded.strides[-1] * hop_size,
    )
    windows = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)

    # Apply window function and compute FFT
    spec = fft.rfft(windows * window[:, None], fft_size, norm="backward", axis=-2)

    # move stft dims to the front (it's what the tv conv expects)
    if stft_dims_first:
        spec = np.moveaxis(np.moveaxis(spec, -2, 0), -1, 0)  # (frames, freq, ...)
    spec = np.ascontiguousarray(spec)

    return spec


def generate_interpolation_matrix(
    ir_times: np.ndarray,
    sr: custom_types.Numeric = config.SAMPLE_RATE,
    hop_size: custom_types.Numeric = config.HOP_SIZE,
    n_frames: Optional[custom_types.Numeric] = None,
) -> np.ndarray:
    """
    Generate impulse response interpolation weights that determines how the source moves through space.

    This is currently simple linear interpolation between ir_times.

    Arguments:
        ir_times (np.ndarray): The IR start times.
        sr (int): Samples per second.
        hop_size (int): The stft hop size.
        n_frames (int): The number of stft frames. Defaults to the maximum frame in ir_times.

    Returns:
        np.ndarray: the interpolation weights.
    """
    # frame indices when each IR starts: (n_irs)
    frames = np.round((ir_times * sr + hop_size) / hop_size)
    n_frames = n_frames if n_frames is not None else int(frames[-1])

    # IR interpolation weights between 0 and 1: (n_frames, n_irs)
    g_interp = np.zeros((n_frames, len(frames)))

    for ni in range(len(frames) - 1):
        tpts = np.arange(frames[ni], frames[ni + 1] + 1, dtype=int) - 1
        ntpts_ratio = np.linspace(0, 1, len(tpts))
        g_interp[tpts, ni] = 1 - ntpts_ratio
        g_interp[tpts, ni + 1] = ntpts_ratio

    return g_interp


def perform_time_variant_convolution(
    s_audio: np.ndarray,
    s_ir: np.ndarray,
    w_ir: np.ndarray,
    ir_slice_min: custom_types.Numeric = 0,
    ir_relevant_ratio_max: custom_types.Numeric = 0.5,
) -> np.ndarray:
    """
    Convolve a bank of time-varying impulse responses with an audio spectrogram.

    Arguments:
        s_audio (np.ndarray): Input audio spectrogram with shape (frames, frequency).
        s_ir (np.ndarray): Input impulse response spectrograms with shape (frames, frequency, channels, # of IRs).
        w_ir (np.ndarray): Impulse response mixing weights between [0, 1] with shape (frames, # of IRs).
        ir_slice_min (Numeric): The minimum number of IRs where it makes sense to start optimizing the matrix
            multiplication by copying the non-zero subset of the `w_ir` matrix
        ir_relevant_ratio_max (Numeric): decides if the matrix should be subselected based on the proportion of IRs
            that are active.

    Returns:
        np.ndarray: the convolved audio spectrogram with shape (frames, frequency).
    """
    # get shapes
    n_frames_ir, n_freq, n_ch, n_irs = s_ir.shape
    n_frames = min(
        s_audio.shape[0], w_ir.shape[0]
    )  # TODO: constant pad ir_interp to sigspec length

    # Invert time for convolution
    s_audio = np.ascontiguousarray(s_audio[::-1])
    w_ir = np.ascontiguousarray(w_ir[::-1]).astype(complex)

    # Output: spatialized stft
    spatial_stft = np.empty((n_frames, n_freq, n_ch), dtype=complex)

    for i in range(n_frames):
        # slice time window that IRs are defined over
        i_ir = -i - 1
        j_ir = min(i_ir + n_frames_ir, 0) or None
        sir = s_ir[: i + 1]
        wir = w_ir[i_ir:j_ir]
        s = s_audio[i_ir:j_ir]

        # drop inactive irs to reduce computation
        # _ir_slice_min refers to the minimum number of IRs where it makes sense
        # to start optimizing the matrix multiplication by copying the non-zero
        # subset of the W_ir matrix.
        if ir_slice_min is not None and n_irs >= ir_slice_min:
            relevant = np.any(wir != 0, axis=0)
            # _ir_relevant_ratio_max decides if the matrix should be subselected
            # based on the proportion of IRs that are active. Basically, if all IRs
            # have some non-zero weight, then there is no point in copying the matrix.
            if relevant.mean() < ir_relevant_ratio_max:  # could optimize this
                sir = sir[
                    :, :, :, relevant
                ]  # this is a copy because of the boolean array :/
                wir = wir[:, relevant]

        # compute the weighted IR spectrogram
        # (frame, freq, ch, nir) x (frame, _, _, nir) = (frame, freq, ch, _)
        ctf_ltv = np.einsum("ijkl,il->ijk", sir, wir)

        # Multiply the signal spectrogram with the CTF
        # (frame, freq, ch) x (frame, freq, _) = (freq, ch)
        Si = np.einsum("ijk,ij->jk", ctf_ltv, s)

        spatial_stft[i] = Si  # (frame, freq, ch)

    return spatial_stft


def istft_overlap_synthesis(
    spatial_stft: np.ndarray,
    fft_size: custom_types.Numeric = config.FFT_SIZE,
    win_size: custom_types.Numeric = config.WIN_SIZE,
    hop_size: custom_types.Numeric = config.HOP_SIZE,
) -> np.ndarray:
    """
    Given a stft, recompose it into audio samples using overlap-add synthesis.
    """
    n_frames, _, n_ch = spatial_stft.shape

    # Inverse FFT
    audio_frames = np.real(fft.irfft(spatial_stft, n=fft_size, axis=1, norm="forward"))

    # Overlap-add synthesis for all frames
    spatial_audio = np.zeros(((n_frames + 1) * hop_size + win_size, n_ch))
    for i in range(n_frames):
        spatial_audio[i * hop_size : i * hop_size + fft_size] += audio_frames[i]

    return spatial_audio[win_size : n_frames * hop_size, :]


def time_variant_convolution(
    irs: np.ndarray,
    event: Event,
    fft_size: Optional[custom_types.Numeric] = config.FFT_SIZE,
    win_size: Optional[custom_types.Numeric] = config.WIN_SIZE,
    hop_size: Optional[custom_types.Numeric] = config.HOP_SIZE,
) -> np.ndarray:
    """
    Performs time-variant convolution for given IRs and Event object
    """

    # Grab the event audio (this should already be loaded)
    audio = event.load_audio()

    # Get parameters and shapes
    win_size = utils.sanitise_positive_number(win_size, cast_to=int)
    hop_size = utils.sanitise_positive_number(hop_size, cast_to=int)
    fft_size = utils.sanitise_positive_number(fft_size, cast_to=int)

    # Compute the spectrograms for both the IRs and the audio
    # Output is (n_frames, n_freq_bins, n_capsules, n_sources)
    ir_spec = stft(irs, fft_size, win_size, hop_size)
    audio_spec = stft(audio, fft_size, win_size, hop_size)

    # Interpolate between the duration of the audio file and the number of IRs to get the time matrix
    ir_times = np.linspace(0, event.duration, len(event))
    w_ir = generate_interpolation_matrix(ir_times, event.sample_rate, hop_size)

    # Convolve signal with irs
    # Output is (n_audio_frames, freq_bins, n_capsules)
    spatial_stft = perform_time_variant_convolution(audio_spec, ir_spec, w_ir)

    # Output is (n_channels, n_samples)
    return istft_overlap_synthesis(spatial_stft, fft_size, win_size, hop_size).T


# noinspection PyProtectedMember
def generate_scene_audio_from_events(scene: Scene) -> None:
    """
    Generate complete audio from a scene, including all events and any background noise

    Note that this function has no direct return. Instead, the finalised audio is written to the `Scene` object
    as an attribute, and can be saved with (for instance) `librosa` or `soundfile`.

    Returns:
        None
    """
    # We'll create audio separately for every microphone added to the state
    for mic_alias in scene.state.microphones.keys():

        # Create empty array with shape (n_channels, n_samples)
        channels = max(
            [ev.spatial_audio[mic_alias].shape[0] for ev in scene.events.values()]
        )
        duration = round(scene.duration * scene.sample_rate)
        scene_audio = np.zeros((channels, duration), dtype=np.float32)

        # If we have ambient noise for the scene, and it is valid, add it in now
        if len(scene.ambience) > 0:
            for ambience in scene.ambience.values():
                if not isinstance(ambience, Ambience):
                    raise TypeError(
                        f"Expected scene ambient noise to be of type Ambience, but got {type(ambience)}!"
                    )

                ambient_noise = ambience.load_ambience(normalize=True)
                if ambient_noise.shape != scene_audio.shape:
                    raise ValueError(
                        f"Scene ambient noise does not match expected shape. "
                        f"Expected {scene_audio.shape}, but got {ambient_noise.shape}."
                    )

                # Now we scale to match the desired noise floor (taken from SpatialScaper)
                scaled = db_to_multiplier(
                    ambience.ref_db, np.mean(np.abs(ambient_noise))
                )
                amb = scaled * ambient_noise

                # TODO: ideally, we can also support adding noise with a given offset and duration
                scene_audio += amb

        # Iterate over all the events
        for event in scene.events.values():
            # Compute scene time in samples
            scene_start = max(0, round(event.scene_start * scene.sample_rate))
            scene_end = min(round(event.scene_end * scene.sample_rate), duration)

            # Ensure valid slice
            if scene_end <= scene_start:
                logger.warning(
                    f"Skipping event due to invalid slice: start={scene_start}, end={scene_end}"
                )
                continue

            # Truncate or pad spatial_audio to fit the scene slice length
            num_samples = scene_end - scene_start
            spatial_audio = utils.pad_or_truncate_audio(
                event.spatial_audio[mic_alias], num_samples
            )

            # Additive synthesis
            scene_audio[:, scene_start:scene_end] += spatial_audio

            # Additionally, set the "padded" audio in the Event
            event_padded_audio = np.zeros_like(scene_audio)
            event_padded_audio[:, scene_start:scene_end] += spatial_audio
            event._spatial_audio_padded[mic_alias] = event_padded_audio

            # Additionally, set pad "dry" audio if required
            if event._spatial_audio_dry.get(mic_alias) is not None:
                event_dry_padded = np.zeros(
                    scene_audio.shape[1], dtype=scene_audio.dtype
                )
                # Little hack with the indexing: audio goes mono -> stereo -> mono
                dry_audio_padded = utils.pad_or_truncate_audio(
                    event._spatial_audio_dry[mic_alias][None, :], num_samples
                )[0]
                event_dry_padded[scene_start:scene_end] += dry_audio_padded
                event._spatial_audio_dry_padded[mic_alias] = event_dry_padded

        # Sanity check everything
        librosa.util.valid_audio(scene_audio)
        utils.validate_shape(scene_audio.shape, (channels, duration))

        scene.audio[mic_alias] = scene_audio


def normalize_irs(irs: np.ndarray) -> np.ndarray:
    """
    Normalizes impulse responses based on their energy.

    This function calculates the energy of each impulse response as the square root of the sum of its squared values.
    It then normalizes each impulse response by the mean energy across all responses.

    Taken from [`SpatialScaper`](https://github.com/marl/SpatialScaper/blob/main/spatialscaper/spatialize.py#L193)

    Parameters:
        irs (numpy.array): A 2D or 3D array of impulse responses, where each row (in the 2D case) or matrix
            (in the 3D case) represents an individual impulse response.

    Returns:
        numpy.array: The normalized impulse responses, having the same shape as the input `IRs`.

    Example:
        # Example of normalizing a set of impulse responses
        impulse_responses = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        normalized_IRs = IR_normalizer(impulse_responses)
    """
    e = np.sqrt(np.sum(np.power(np.abs(irs), 2), axis=-1, keepdims=True))
    # Prevents divide by zero
    e += utils.tiny(e)
    return irs / np.mean(e, axis=-2, keepdims=True)


# noinspection PyProtectedMember
def compute_dry_audio(
    event: Event, irs: np.ndarray, event_scale: float, mic_alias: str
) -> None:
    """
    Computes and stores the dry (direct-path) audio for an Event.

    The dry audio represents only the direct path and early reflections of the event’s sound as captured by a specific
    reference channel in the provided impulse responses (IRs). It is obtained by truncating the IR around its peak
    (the direct path) using a specified time window (`direct_path_time_ms`) and then convolving the dry event audio with
    this truncated IR. The resulting signal is scaled by `event_scale` and stored as a spatialized channel for the
    given `mic_alias`.

    If neither `ref_ir_channel` nor `direct_path_time_ms` are provided in the Event, dry audio is not computed (default
    behavior). If only one of these parameters is provided, a warning is logged and computation is skipped.

    See https://github.com/nttcslab/dcase2025_task4_baseline/blob/12063c5219cab094826e839be9a9bd095a6b441c/src/modules/spatialscaper2/semseg_spatialscaper.py#L73

    Arguments:
        event (Event): The Event instance containing metadata and access to the raw audio signal.
        irs (np.ndarray): Array of impulse responses with shape `(n_channels, n_irs, n_ir_samples)`.
        event_scale (float): Scaling factor (derived from target reference dB and SNR) applied to the dry audio.
        mic_alias (str): Key or alias under which to store the computed dry audio in `event._spatial_audio_dry`.

    Raises:
        ValueError: If the specified `ref_ir_channel` index is out of range for the given IRs.

    Returns:
        None: The computed and scaled dry audio is stored internally in `event._spatial_audio_dry[mic_alias]`.
    """

    # We don't want to compute dry audio (default behaviour)
    if event.ref_ir_channel is None and event.direct_path_time_ms is None:
        return

    # We want to compute dry audio
    elif event.ref_ir_channel is not None and event.direct_path_time_ms is not None:
        # Get reference channel for the IR and raise an error if invalid
        ref_channel = event.ref_ir_channel
        if ref_channel > irs.shape[0]:
            raise ValueError(
                f"Reference channel index out of range for IRs with {irs.shape[0]} channels"
            )

        # See https://github.com/nttcslab/dcase2025_task4_baseline/blob/12063c5219cab094826e839be9a9bd095a6b441c/src/modules/spatialscaper2/semseg_spatialscaper.py#L73
        # Get direct path time, convert to samples, and raise an error if invalid
        low, high = event.direct_path_time_ms
        low_sp = int(low * event.sample_rate / 1000)
        high_sp = int(high * event.sample_rate / 1000)

        # Find peak of IR
        peak = np.argmax(irs[ref_channel, 0, :], axis=0)
        ir_direct_path = irs[ref_channel, 0, :].copy()
        if peak + high_sp < ir_direct_path.shape[0]:
            ir_direct_path[peak + high_sp :] = 0
        if peak - low_sp > 0:
            ir_direct_path[: peak - low_sp] = 0

        # Do the convolution
        dry = signal.fftconvolve(
            event.load_audio(ignore_cache=False), ir_direct_path, mode="full", axes=0
        )
        dry_scaled = dry * event_scale

        # Store as an attribute of the Event
        event._spatial_audio_dry[mic_alias] = dry_scaled

    # We've made a mistake and not provided both arguments, so log and return
    else:
        logger.warning(
            "Only one of `ref_ir_channel` or `direct_path_time` were specified when creating the Event. "
            "Dry audio will not be computed for this Event. Pass both variables to compute dry audio."
        )
        return


def render_event_audio(
    event: Event,
    irs: np.ndarray,
    mic_alias: str,
    ref_db: custom_types.Numeric = config.DEFAULT_REF_DB,
    ignore_cache: Optional[bool] = True,
    fft_size: Optional[custom_types.Numeric] = config.FFT_SIZE,
    win_size: Optional[custom_types.Numeric] = config.WIN_SIZE,
    hop_size: Optional[custom_types.Numeric] = config.HOP_SIZE,
) -> None:
    """
    Renders audio for a given `Event` object.

    Audio is rendered following the following stages:
        - Load audio for the `Event` object and transform according to given SNR, noise floor, effects, etc.
        - Convolve `Event` audio with IRs from associated `Emitter` objects

    Note that this function has no direct return. Instead, it simply populates the `spatial_audio` attribute of the
    event, which can then be written using (e.g.) `librosa`, `soundfile`, etc.

    Arguments:
        event (Event): the Event object to render audio for
        irs (np.ndarray): the IR audio array for the given event, taken from the WorldState and this event's Emitters
        mic_alias: the microphone alias associated with the IRs used here
        ref_db (custom_types.Numeric): the noise floor for the Scene
        ignore_cache (bool): if True, any cached spatial audio from a previous call to this function will be discarded
        fft_size (custom_types.Numeric): size of the FFT, defaults to 512 samples
        win_size (int): the window size of the FFT, defaults to 256 samples
        hop_size (custom_types.Numeric): the size of the hop between FFTs, defaults to 128 samples

    Returns:
        None
    """
    # In cases where we've already cached the spatial audio, and we want to use it, skip over
    if mic_alias in event.spatial_audio.keys() and not ignore_cache:
        return

    # Grab the IRs for the current event's emitters (make a copy first)
    #  This gets us (N_capsules, N_emitters, N_samples)
    irs_copy = irs.copy()
    n_ch, n_emitters, n_ir_samples = irs_copy.shape

    # Grab the audio for the event as well and validate
    #  This also applies any augmentations, etc. associated with the Event and normalizes the audio
    audio = event.load_audio(ignore_cache=ignore_cache, normalize=True)
    librosa.util.valid_audio(audio)
    n_audio_samples = audio.shape[0]

    # Normalize IRs for this event
    #  This goes (n_caps, n_source, n_samp) -> (n_source, n_caps, n_samp)
    #  Then normalization is applied to each (n_caps, n_samp)
    #  Stacked back to (n_source, n_caps, n_samp)
    #  Then transposed back to (n_caps, n_source, n_samp)
    irs_copy = normalize_irs(irs_copy.transpose(1, 0, 2)).transpose(1, 0, 2)

    # THIS IS NOW EQUIVALENT TO spatialscaper.spatialize FUNC
    # Only a single emitter (IR): we can convolve easily with scipy
    if n_emitters == 1:
        if event.is_moving:
            raise ValueError(
                "Moving Event has only one emitter!"
            )  # something has gone very wrong to hit this
        spatial = time_invariant_convolution(audio, irs_copy[:, 0].T)

    # No emitters: means that audio is not spatialized
    elif n_emitters == 0:
        logger.warning(
            f"No IRs were found for Event with alias {event.alias}. Audio is being tiled along the "
            f"channel dimension to match the expected shape {n_ch, n_audio_samples}."
        )
        spatial = np.repeat(audio[:, None], n_ch, 1).T

    # Moving sound sources: need to do time-variant convolution
    else:
        if not event.is_moving:
            raise ValueError(
                "Expected a moving event!"
            )  # something has gone very wrong to hit this
        spatial = time_variant_convolution(
            irs_copy, event, fft_size, win_size, hop_size
        )

    # Pad or truncate the audio to match the desired number of samples
    spatial = utils.pad_or_truncate_audio(spatial, n_audio_samples)

    # Deal with amplitude: this logic is taken from SpatialScaper
    #  This scales the audio simply to the maximum SNR
    spatial = apply_snr(spatial, event.snr)
    # SPATIAL IS NOW EQUIVALENT TO OUTPUT OF SPATIALSCAPER FUNC

    #  This scales to match the Scene's noise floor + the SNR for the Event
    event_scale = db_to_multiplier(ref_db + event.snr, np.mean(np.abs(spatial)))
    spatial = event_scale * spatial

    # Validate that the spatial audio has the expected shape and it is valid audio
    utils.validate_shape(spatial.shape, (n_ch, n_audio_samples))
    librosa.util.valid_audio(spatial)

    # Cast the audio as an attribute of the event
    event.spatial_audio[mic_alias] = spatial

    compute_dry_audio(event, irs_copy, event_scale, mic_alias)

    # Function has no direct return


def render_audio_for_all_scene_events(
    scene: Scene, ignore_cache: Optional[bool] = False
) -> None:
    """
    Renders audio for all `Events` associated with a given `Scene` object.

    Audio is rendered following the following stages:
        - Generate IRs for associated microphones and emitters inside `Scene.WorldState` using ray-tracing engine
        - Load audio for all `Scene.Event` objects and transform according to given SNR, effects, etc.
        - Convolve `Scene.Event` audio with IRs from associated `Emitter` objects

    Note that this function has no direct return. Instead, the audio for every `Event` object is populated inside the
    `Event.spatial_audio` attribute, which can be written using (for example) `pysoundfile` or `librosa`.

    Arguments:
        scene: Scene object with associated `WorldState`, `Emitter`, `MicArray`, `Event` objects added.
        ignore_cache (optional): If True, cached Event audio will be ignored

    Returns:
        None
    """

    # If we're invalidating the cache, always re-simulate the IRs whenever calling this function
    if ignore_cache:
        scene.state.simulate()
    # Otherwise, only run the synthesis if this hasn't already been done
    else:
        try:
            _ = scene.state.irs
        except AttributeError:
            scene.state.simulate()

    # Validate the scene
    validate_scene(scene)
    # Grab the IRs from the WorldState
    #  This is a dictionary with format {mic000: [N_channels, N_emitters, N_samples], ...}
    irs = scene.state.get_irs()

    # Iterate over all microphones
    start = time()
    for mic_alias, mic_ir in irs.items():
        # We need a separate counter for each microphone
        emitter_counter = 0

        # Iterate over all events
        for event_alias, event in scene.events.items():

            # Grab the IRs for the current event's emitters
            #  This gets us (N_capsules, N_emitters, N_samples)
            event_irs = mic_ir[:, emitter_counter : len(event) + emitter_counter, :]

            # Render the audio for the event at this microphone
            #  This function has no return, instead it just sets the `spatial_audio` attribute for the Event
            render_event_audio(
                event,
                event_irs,
                mic_alias=mic_alias,
                ref_db=scene.ref_db,
                ignore_cache=ignore_cache,
            )

            # Update the counter
            emitter_counter += len(event)

    logger.info(f"Rendered scene audio in {(time() - start):.2f} seconds!")


# noinspection PyProtectedMember
def validate_scene(scene: Scene) -> None:
    """
    Validates a Scene object before synthesis and raises errors as required.

    Returns:
        None
    """
    # Validate WorldState
    if scene.state.num_emitters == 0:
        raise ValueError("WorldState has no emitters!")
    if len(scene.state.microphones) == 0:
        raise ValueError("WorldState has no microphones!")

    # Validate Events
    if len(scene.events) == 0:
        raise ValueError("Scene has no events!")

    # Count number of emitters registered to all Events
    total_ems = 0
    for alias, ev in scene.events.items():
        # Raise an error if any events have no emitters registered
        try:
            total_ems += len(ev)
        except ValueError:
            raise ValueError(
                f"Event with alias '{alias}' has no emitters registered. Has it been orphaned?"
            )

    # Remaining checks only apply to ray-tracing contexts
    if not scene.state.name.upper() == "RLR":
        return

    # Validate ray-tracing engine
    if scene.state.ctx.get_listener_count() == 0:
        raise ValueError("Ray-tracing engine has no listeners!")
    if scene.state.ctx.get_source_count() == 0:
        raise ValueError("Ray-tracing engine has no sources!")

    # Validate across all parts of the library, e.g. WorldState, Scene, ray-tracing engine
    vals = (
        total_ems,
        scene.state.num_emitters,
        scene.state.ctx.get_source_count(),
    )
    if not all(v == vals[0] for v in vals):
        raise ValueError(
            f"Mismatching number of emitters, events, and sources! "
            f"Got {len(scene.events)} events, {scene.state.num_emitters} emitters, "
            f"{scene.state.ctx.get_source_count()} sources. "
            f"Have any been orphaned?"
        )

    capsules = sum(m.n_listeners for m in scene.state.microphones.values())
    if capsules != scene.state.ctx.get_listener_count():
        raise ValueError(
            f"Mismatching number of microphones and listeners! "
            f"Got {capsules} capsules, {scene.state.ctx.get_listener_count()} listeners. "
            f"Have any been orphaned?"
        )


def generate_dcase2024_metadata(scene: Scene) -> dict[str, pd.DataFrame]:
    """
    Given a Scene, generate metadata for each microphone in the DCASE 2024 format.

    The output format is given as {"mic_alias_0": <pd.DataFrame>, "mic_alias_1": <pd.DataFrame>} for every microphone
    added to the scene. The exact specification of the metadata can be found on the [DCASE 2024 challenge website]
    (https://dcase.community/challenge2024/task-audio-and-audiovisual-sound-event-localization-and-detection-with-source-distance-estimation)

    In particular, the columns of each dataframe are as follows:
    - frame number (int): the index of the frame
    - active class index (int): the index of the soundevent: see `audiblelight.event.DCASE_SOUND_EVENT_CLASSES` for
        a complete mapping.
    - source number index (int): a unique integer identifier for each event in the scene.
    - azimuth (int): the azimuth, increasing counter-clockwise (ϕ=90∘ at the left, ϕ=0∘ at the front).
    - elevation (int): the elevation angle (θ=0∘ at the front).
    - distance (int): the distance from the microphone, measured in centimeters.

    The audio is quantised to 10 frames per second (i.e., frame length = 100 ms). In cases of moving trajectories, the
    position of each IR is linearly interpolated throughout the duration of the audio file in order to obtain a value
    for azimuth, elevation, and distance estimated at every frame.

    Note that, `source number index` value is assigned **separately** for each class (in the STARSS format):
    thus, with two `telephone` classes and one `femaleSpeech`, we would expect to see values of 0 and 1 for the two
    `telephone` instances and only `0` for the `femaleSpeech` instance. Events that share the same audio file are
    always assigned the same source ID every time they occur.

    Finally, note that frames without sound events are omitted from the output.
    """

    # Produce an array of frames, lasting as long as the scene itself.
    frames = np.round(np.arange(0, scene.duration + 0.1, 0.1), 1)

    # Aliases for all microphones
    microphones = list(scene.state.microphones.keys())
    res = {mic: [] for mic in microphones}

    # This mapping will be used to count the number of times that each class IDX appears
    unique_ids = Counter()

    # Need to sort events by starting time in the scene
    events = scene.get_events()
    sorted_events = sorted(events, key=lambda e: e.scene_start)

    # Keep track of the files we've already seen and their IDs
    seen_filepaths = {}

    for event in sorted_events:
        # Determine frame indices for event start and end
        start_idx = np.where(frames == round(max(event.scene_start, 0.0), 1))[0][0]
        end_idx = np.where(frames == round(min(event.scene_end, scene.duration), 1))[0][
            0
        ]

        # This is the frame indices where the frame lasts
        event_range = np.arange(start_idx, end_idx + 1)

        # Raise an error if the event has no DCASE class idx or this is somehow invalid otherwise
        if not isinstance(event.class_id, int):
            raise ValueError(
                "Can't convert Event to DCASE format without valid DCASE class indices"
            )

        # If we haven't already seen this file before, grab the ID
        if event.filename not in seen_filepaths:
            source_idx = unique_ids.get(event.class_id, 0)
            seen_filepaths[event.filename] = source_idx
            # Increment the counter by one for the next occurrence of this class
            unique_ids[event.class_id] += 1

        # If we have seen this file before, use the same source ID as before
        else:
            source_idx = seen_filepaths[event.filename]

        # Iterate over every microphone for each event
        for mic in microphones:
            # Processing static events
            if not event.is_moving:
                # Static events just have one emitter, so grab the relative position to the mic directly
                az, elv, dist = event.emitters[0].coordinates_relative_polar[mic][0]
                # Need to round values and convert metres -> centimetres
                az, elv, dist = round(az), round(elv), round(dist * 100)
                # We want a new row for every frame
                frame_data = [
                    [int(idx), event.class_id, source_idx, az, elv, dist]
                    for idx in event_range
                ]
                res[mic].extend(frame_data)

            # Processing moving events
            else:
                # Get the relative positions of all emitters for this event vs the current mic
                coords = np.vstack(
                    [e.coordinates_relative_polar[mic] for e in event.emitters]
                )
                # Get the times we'll interpolate using
                interp_times = frames[event_range]
                # Get the approximate times for every coordinate
                coord_times = np.linspace(
                    min(interp_times), max(interp_times), num=len(coords)
                )
                # Interpolate between the coordinates and timepoints
                interpolated = np.stack(
                    [
                        np.interp(interp_times, coord_times, coords[:, dim])
                        for dim in range(coords.shape[1])
                    ],
                    axis=1,
                )
                # Create a separate row for each frame
                for idx, (az, elv, dist) in zip(event_range, interpolated):
                    # Need to round values and convert metres -> centimetres
                    az, elv, dist = round(az), round(elv), round(dist * 100)
                    frame_data = [
                        int(idx),
                        event.class_id,
                        source_idx,
                        az,
                        elv,
                        dist,
                    ]
                    res[mic].append(frame_data)

    # Output dataframes
    res_df = {mic: None for mic in microphones}
    for mic, data in res.items():
        res_df[mic] = (
            pd.DataFrame(data, columns=DCASE_2024_COLUMNS)
            .sort_values(["frame_number", "active_class_index", "source_number_index"])
            .set_index("frame_number")
        )

    return res_df
