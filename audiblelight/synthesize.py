#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules and functions for "synthesizing" outputs (including audio, video, image, metadata)"""

from time import time
from typing import Optional

import librosa
import numpy as np
from loguru import logger
from scipy import fft, signal
from tqdm import tqdm

from audiblelight import utils
from audiblelight.ambience import Ambience
from audiblelight.core import Scene
from audiblelight.event import Event


def apply_snr(x: np.ndarray, snr: utils.Numeric) -> np.ndarray:
    """
    Scale an audio signal to a given maximum SNR.

    Taken from [`SpatialScaper`](https://github.com/marl/SpatialScaper/blob/dd130d1e0f8aef0c93f5e1b73c3445f855b92e7b/spatialscaper/spatialize.py#L147)

    Return:
        np.ndarray: the scaled audio signal
    """
    return x * snr / np.abs(x).max(initial=1e-15)


def db_to_multiplier(db: utils.Numeric, x: utils.Numeric) -> float:
    """
    Calculates the multiplier factor from a decibel (dB) value that, when applied to x, adjusts its amplitude to
    reflect the specified dB. The relationship is based on the formula 20 * log10(factor * x) ≈ db.

    Taken from [`SpatialScaper`](https://github.com/marl/SpatialScaper/blob/dd130d1e0f8aef0c93f5e1b73c3445f855b92e7b/spatialscaper/utils.py#L287)

    Arguments:
        db (float): The target decibel change to be applied.
        x  (float): The original amplitude of x

    Returns:
        float: The multiplier factor.
    """
    return 10 ** (db / 20) / x


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

    # Get all the shapes
    n_audio_samples, n_audio_channels = audio.shape
    n_ir_samples, n_ir_channels = ir.shape

    # Pad out the IR signal if it is shorter than the audio, and raise a warning
    if n_ir_samples < n_audio_samples:
        logger.warning(
            f"IR has fewer samples than audio (IR: {n_ir_samples}, audio: {n_audio_samples}). "
            f"IR will be right-padded with zeros to match audio length!"
        )
        ir = np.pad(
            ir,
            ((0, n_audio_samples - n_ir_samples), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # Perform the convolution using FFT method
    #  the output shape will be (n_ir_samples, n_ir_channels)
    convolve: np.ndarray = signal.fftconvolve(audio, ir, mode="full", axes=0)

    # Transpose so we get (n_channels, n_samples)
    return convolve.T


def stft(
    y: np.ndarray,
    fft_size: utils.Numeric = 512,
    win_size: utils.Numeric = 256,
    hop_size: utils.Numeric = 128,
    stft_dims_first: bool = True,
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
    sr: utils.Numeric,
    hop_size: utils.Numeric,
    n_frames: Optional[utils.Numeric] = None,
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
    ir_slice_min: utils.Numeric = 0,
    ir_relevant_ratio_max: utils.Numeric = 0.5,
) -> np.ndarray:
    """
    Convolve a bank of time-varying impulse responses with an audio spectrogram.

    Arguments:
        s_audio (np.ndarray): Input audio spectrogram with shape (frames, frequency).
        s_ir (np.ndarray): Input impulse response spectrograms with shape (frames, frequency, channels, # of IRs).
        w_ir (np.ndarray): Impulse response mixing weights between [0, 1] with shape (frames, # of IRs).

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
    fft_size: utils.Numeric,
    win_size: utils.Numeric,
    hop_size: utils.Numeric,
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
    win_size: utils.Numeric,
    hop_size: Optional[utils.Numeric] = None,
) -> np.ndarray:
    """
    Performs time-variant convolution for given IRs and Event object
    """

    # Grab the event audio (this should already be loaded)
    audio = event.load_audio()

    # Get parameters and shapes
    win_size = int(utils.sanitise_positive_number(win_size))
    if hop_size is None:
        hop_size = win_size // 2
    hop_size = int(utils.sanitise_positive_number(hop_size))
    fft_size = 2 * win_size

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


def generate_scene_audio_from_events(scene: Scene) -> None:
    """
    Generate complete audio from a scene, including all events and any background noise

    Note that this function has no direct return. Instead, the finalised audio is written to the `Scene` object
    as an attribute, and can be saved with (for instance) `librosa` or `soundfile`.

    Returns:
        None
    """
    # Create empty array with shape (n_channels, n_samples)
    channels = max([ev.spatial_audio.shape[0] for ev in scene.events.values()])
    duration = round(scene.duration * scene.sample_rate)
    scene_audio = np.zeros((channels, duration), dtype=np.float32)

    # If we have ambient noise for the scene, and it is valid, add it in now
    if len(scene.ambience) > 0:
        for ambience in scene.ambience.values():
            if not isinstance(ambience, Ambience):
                raise TypeError(
                    f"Expected scene ambient noise to be of type Ambience, but got {type(ambience)}!"
                )

            ambient_noise = ambience.load_ambience()
            if ambient_noise.shape != scene_audio.shape:
                raise ValueError(
                    f"Scene ambient noise does not match expected shape. "
                    f"Expected {scene_audio.shape}, but got {ambient_noise.shape}."
                )

            # Now we scale to match the desired noise floor (taken from SpatialScaper)
            scaled = db_to_multiplier(ambience.ref_db, np.mean(np.abs(ambient_noise)))
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
        spatial_audio = pad_or_truncate_audio(event.spatial_audio, num_samples)

        # Additive synthesis
        scene_audio[:, scene_start:scene_end] += spatial_audio

    # Sanity check everything
    librosa.util.valid_audio(scene_audio)
    utils.validate_shape(scene_audio.shape, (channels, duration))

    scene.audio = scene_audio


def pad_or_truncate_audio(
    audio: np.ndarray, desired_samples: utils.Numeric
) -> np.ndarray:
    """
    Pads or truncates audio with desired number of samples.
    """
    # Audio is too short, needs padding
    if audio.shape[1] < desired_samples:
        return np.pad(
            audio, ((0, 0), (0, desired_samples - audio.shape[1])), mode="constant"
        )
    # Audio is too long, needs truncating
    elif audio.shape[1] > desired_samples:
        return audio[:, :desired_samples]
    # Audio is just right
    else:
        return audio


def render_event_audio(
    event: Event,
    irs: np.ndarray,
    ref_db: utils.Numeric,
    ignore_cache: bool = True,
    win_size: utils.Numeric = 512,
    hop_size: Optional[utils.Numeric] = None,
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
        ref_db (utils.Numeric): the noise floor for the Scene
        ignore_cache (bool): if True, any cached spatial audio from a previous call to this function will be discarded
        win_size (int): the window size of the FFT, defaults to 512 samples
        hop_size (utils.Numeric): the size of the hop between FFTs, defaults to (win_size // 2) samples

    Returns:
        None
    """
    # In cases where we've already cached the spatial audio, and we want to use it, skip over
    if event.spatial_audio is not None and not ignore_cache:
        return

    # Grab the IRs for the current event's emitters
    #  This gets us (N_capsules, N_emitters, N_samples)
    n_ch, n_emitters, n_ir_samples = irs.shape

    # Grab the audio for the event as well and validate
    audio = event.load_audio(ignore_cache=ignore_cache)
    librosa.util.valid_audio(audio)

    # TODO: this is when we'd also apply any data augmentation, etc.
    n_audio_samples = audio.shape[0]

    # Only a single emitter (IR): we can convolve easily with scipy
    if n_emitters == 1:
        if event.is_moving:
            raise ValueError(
                "Moving Event has only one emitter!"
            )  # something has gone very wrong to hit this
        # TODO: if any emitters are not mono, this will break silently
        spatial = time_invariant_convolution(audio, irs[:, 0].T)

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
        spatial = time_variant_convolution(irs, event, win_size, hop_size)

    # Pad or truncate the audio to match the desired number of samples
    spatial = pad_or_truncate_audio(spatial, n_audio_samples)

    # Deal with amplitude: this logic is taken from SpatialScaper
    #  This scales the audio simply to the maximum SNR
    spatial = apply_snr(spatial, event.snr)
    #  This scales to match the Scene's noise floor + the SNR for the Event
    event_scale = db_to_multiplier(ref_db + event.snr, np.mean(np.abs(spatial)))
    spatial = event_scale * spatial

    # Validate that the spatial audio has the expected shape and it is valid audio
    utils.validate_shape(spatial.shape, (n_ch, n_audio_samples))
    librosa.util.valid_audio(spatial)

    # Cast the audio as an attribute of the event, function has no direct return
    event.spatial_audio = spatial


def render_audio_for_all_scene_events(scene: Scene, ignore_cache: bool = True) -> None:
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
        ignore_cache (optional): If True (default), cached Event audio will be ignored

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

    # Grab the IRs from the entire WorldState
    #  The expected IR shape is (N_capsules, N_emitters, N_channels (== 1), N_samples)
    irs = scene.state.ctx.get_audio()
    # TODO: probably won't work with more than one microphone!
    emitter_counter = 0

    # Iterate over each one of our Events
    start = time()
    for event_alias, event in tqdm(
        scene.events.items(), desc="Rendering event audio..."
    ):

        # Grab the IRs for the current event's emitters
        #  This gets us (N_capsules, N_emitters, N_samples)
        event_irs = irs[
            :, emitter_counter : len(event.emitters) + emitter_counter, 0, :
        ]

        # Render the audio for the event
        #  This function has no return, instead it just sets the `spatial_audio` attribute for the Event
        render_event_audio(
            event, event_irs, ref_db=scene.ref_db, ignore_cache=ignore_cache
        )

        # Update the counter
        emitter_counter += len(event.emitters)

    logger.info(f"Rendered scene audio in {(time() - start):.2f} seconds.!")


# noinspection PyProtectedMember
def validate_scene(scene: Scene) -> None:
    """
    Validates a Scene object before synthesis and raises errors as required.

    Returns:
        None
    """
    # Validate WorldState
    if len(scene.state.emitters) == 0:
        raise ValueError("WorldState has no emitters!")
    if len(scene.state.microphones) == 0:
        raise ValueError("WorldState has no microphones!")

    # Validate ray-tracing engine
    if scene.state.ctx.get_listener_count() == 0:
        raise ValueError("Ray-tracing engine has no listeners!")
    if scene.state.ctx.get_source_count() == 0:
        raise ValueError("Ray-tracing engine has no sources!")

    # Validate Events
    if len(scene.events) == 0:
        raise ValueError("Scene has no events!")

    # Validate across all parts of the library, e.g. WorldState, Scene, ray-tracing engine
    vals = (
        sum(
            len(ev) for ev in scene.events.values()
        ),  # sums number of emitters for every event
        sum(
            len(em) for em in scene.state.emitters.values()
        ),  # sums number of emitters in total for the scene
        scene.state.ctx.get_source_count(),
    )
    if not all(v == vals[0] for v in vals):
        raise ValueError(
            f"Mismatching number of emitters, events, and sources! "
            f"Got {len(scene.events)} events, {len(scene.state.emitters)} emitters, "
            f"{scene.state.ctx.get_source_count()} sources."
        )

    capsules = sum(m.n_capsules for m in scene.state.microphones.values())
    if capsules != scene.state.ctx.get_listener_count():
        raise ValueError(
            f"Mismatching number of microphones and listeners! "
            f"Got {capsules} capsules, {scene.state.ctx.get_listener_count()} listeners."
        )

    if any(not ev.has_emitters for ev in scene.events.values()):
        raise ValueError("Some events have no emitters registered to them!")
