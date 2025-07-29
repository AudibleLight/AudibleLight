#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules and functions for "synthesizing" outputs (including audio, video, image, metadata)"""

from time import time

import numpy as np
import librosa
from loguru import logger
from scipy import signal
from tqdm import tqdm

from audiblelight import utils
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
        raise ValueError(f"Only mono input is supported, but got {audio.ndim} dimensions!")
    if ir.ndim != 2:
        raise ValueError(f"Expected shape of IR should be (n_samples, n_channels), but got ({ir.shape}) instead")

    # Shape of both arrays should be (n_samples, n_channels)
    #  where audio[n_channels] == 1, ir[n_channels] >= 1,
    #  and audio[n_samples] may not equal ir[n_samples]
    audio = np.expand_dims(audio, 1)

    # Get all the shapes
    n_audio_samples, n_audio_channels = audio.shape
    n_ir_samples, n_ir_channels = ir.shape

    # Pad out the IR signal if it is shorter than the audio, and raise a warning
    if n_ir_samples < n_audio_samples:
        logger.warning(f"IR has fewer samples than audio (IR: {n_ir_samples}, audio: {n_audio_samples}). "
                       f"IR will be right-padded with zeros to match audio length!")
        ir = np.pad(ir, ((0, n_audio_samples - n_ir_samples), (0, 0)), mode="constant", constant_values=0)

    # Perform the convolution using FFT method
    #  the output shape will be (n_ir_samples, n_ir_channels)
    convolve: np.ndarray = signal.fftconvolve(audio, ir, mode="full", axes=0)
    if convolve.shape[0] > n_audio_samples:
        convolve = convolve[:n_audio_samples, :]

    return convolve


def time_variant_convolution(audio: np.ndarray, ir_matrix: np.ndarray) -> np.ndarray:
    """
    Convolve a bank of time-varying impulse responses with an audio input.
    """
    raise NotImplementedError


def generate_scene_audio_from_events(scene: Scene) -> None:
    """
    Given a `Scene` object, generate a single array that combines audio from all Events, at the correct position.

    Note that this function has no direct return. Instead, the finalised audio is written to the `Scene` object
    as an attribute, and can be saved with (for instance) `librosa` or `soundfile`.

    Returns:
        None
    """
    # Get the sample rate from the ray-tracing engine
    sample_rate = scene.state.ctx.config.sample_rate

    # Create empty array with shape (n_channels, n_samples)
    channels = max([ev.spatial_audio.shape[0] for ev in scene.events.values()])
    duration = round(scene.duration * sample_rate)
    scene_audio = np.zeros((channels, duration), dtype=np.float32)

    # TODO: background noise/ambience gets added here

    # Iterate over all the events
    for event in scene.events.values():
        # Compute scene time in samples
        scene_start = max(0, round(event.scene_start * sample_rate))
        scene_end = min(round(event.scene_end * sample_rate), duration)

        # Ensure valid slice
        if scene_end <= scene_start:
            print(f"Skipping event due to invalid slice: start={scene_start}, end={scene_end}")
            continue

        num_samples = scene_end - scene_start

        # Truncate or pad spatial_audio to fit the scene slice length
        spatial_audio = event.spatial_audio[:, :num_samples]

        # If spatial_audio is shorter than expected, pad it (optional but safe)
        if spatial_audio.shape[1] < num_samples:
            pad_width = num_samples - spatial_audio.shape[1]
            spatial_audio = np.pad(spatial_audio, ((0, 0), (0, pad_width)), mode='constant')

        scene_audio[:, scene_start:scene_end] += spatial_audio

    # Sanity check everything
    librosa.util.valid_audio(scene_audio)
    utils.validate_shape(scene_audio.shape, (channels, duration))

    scene.audio = scene_audio


def render_event_audio(event: Event, irs: np.ndarray, ref_db: utils.Numeric, ignore_cache: bool = True) -> None:
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
            raise ValueError("Moving Event has only one emitter!")  # something has gone very wrong to hit this
        # TODO: if any emitters are not mono, this will break silently
        spatial = time_invariant_convolution(audio, irs[:, 0].T).T

    # No emitters: means that audio is not spatialized
    elif n_emitters == 0:
        logger.warning(f"No IRs were found for Event with alias {event.alias}. Audio is being tiled along the "
                       f"channel dimension to match the expected shape ({n_ch, n_audio_samples}).")
        spatial = np.repeat(audio[:, None], n_ch, 1).T

    # Moving sound sources: need to do time-variant convolution
    else:
        spatial = time_variant_convolution(audio, irs)

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


def render_scene_audio(scene: Scene, ignore_cache: bool = True) -> None:
    """
    Renders audio for a given `Scene` object.

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

    # Try and grab the IRs from the WorldState, or run the synthesis if they're not present
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
    for event_alias, event in tqdm(scene.events.items(), desc="Rendering event audio..."):

        # Grab the IRs for the current event's emitters
        #  This gets us (N_capsules, N_emitters, N_samples)
        event_irs = irs[:, emitter_counter:len(event.emitters) + emitter_counter, 0, :]

        # Render the audio for the event
        #  This function has no return, instead it just sets the `spatial_audio` attribute for the Event
        render_event_audio(event, event_irs, ref_db=scene.ref_db, ignore_cache=ignore_cache)

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
    vals = len(scene.events), len(scene.state.emitters), scene.state.ctx.get_source_count()
    if not all(v == vals[0] for v in vals):
        raise ValueError(f"Mismatching number of emitters, events, and sources! "
                         f"Got {len(scene.events)} events, {len(scene.state.emitters)} emitters, "
                         f"{scene.state.ctx.get_source_count()} sources.")

    capsules = sum(m.n_capsules for m in scene.state.microphones.values())
    if capsules != scene.state.ctx.get_listener_count():
        raise ValueError(f"Mismatching number of microphones and listeners! "
                         f"Got {capsules} capsules, {scene.state.ctx.get_listener_count()} listeners.")
