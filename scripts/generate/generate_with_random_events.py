#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate a simple scene with random moving, static, and ambient sound events and render to audio and JSON."""

import argparse
import json
import os
from pathlib import Path
from time import time

from loguru import logger
from scipy import stats

from audiblelight import config, utils
from audiblelight.core import Scene

# OUTPUT DIRECTORY
OUTFOLDER = utils.get_project_root() / "spatial_scenes"
if not os.path.isdir(OUTFOLDER):
    os.makedirs(OUTFOLDER)

# PATHS
FG_FOLDER = utils.get_project_root() / "tests/test_resources/soundevents"
BG_FOLDER = FG_FOLDER / "domesticSounds"
MESH_PATH = utils.get_project_root() / "tests/test_resources/meshes/Oyens.glb"


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a simple audio scene with a set number of moving and static sound sources in particular positions."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=config.SCENE_DURATION,
        help=f"Duration of the scene in seconds (default: {config.SCENE_DURATION}).",
    )
    parser.add_argument(
        "--n-static",
        type=int,
        default=config.DEFAULT_STATIC_EVENTS,
        help=f"Number of static events to include (default: {config.DEFAULT_STATIC_EVENTS}).",
    )
    parser.add_argument(
        "--n-moving",
        type=int,
        default=config.DEFAULT_MOVING_EVENTS,
        help=f"Number of moving events to include (default: {config.DEFAULT_MOVING_EVENTS}).",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=config.MAX_OVERLAP,
        help=f"Maximum number of overlapping events (default: {config.MAX_OVERLAP}).",
    )
    parser.add_argument(
        "--micarray",
        type=str,
        default=config.MIC_ARRAY_TYPE,
        help=f"Microphone array name or alias (default: {config.MIC_ARRAY_TYPE}).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(OUTFOLDER),
        help=f"Output directory for audio and metadata (default: {str(OUTFOLDER)}).",
    )
    parser.add_argument(
        "--fg-folder",
        type=str,
        default=str(FG_FOLDER),
        help=f"Foreground sound events folder (default: {str(FG_FOLDER)}).",
    )
    parser.add_argument(
        "--bg-folder",
        type=str,
        default=str(BG_FOLDER),
        help=f"Background sound events folder (default: {str(BG_FOLDER)}).",
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=str(MESH_PATH),
        help=f"Path to mesh file (e.g., a building) (default: {str(MESH_PATH)}).",
    )
    parser.add_argument(
        "--ref-db",
        type=float,
        default=config.DEFAULT_REF_DB,
        help=f"Reference decibel level (default: {config.DEFAULT_REF_DB}).",
    )
    parser.add_argument(
        "--min-snr",
        type=int,
        default=config.MIN_EVENT_SNR,
        help=f"Minimum signal-to-noise ratio for placed sound events (default: {config.MIN_EVENT_SNR}).",
    )
    parser.add_argument(
        "--max-snr",
        type=int,
        default=config.MAX_EVENT_SNR,
        help=f"Maximum signal-to-noise ratio for placed sound events (default: {config.MAX_EVENT_SNR}).",
    )
    parser.add_argument(
        "--min-velocity",
        type=float,
        default=config.MIN_EVENT_VELOCITY,
        help=f"Minimum velocity (m/s) for placed sound events (default: {config.MIN_EVENT_VELOCITY}).",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=config.MAX_EVENT_VELOCITY,
        help=f"Maximum velocity (m/s) for placed sound events (default: {config.MAX_EVENT_VELOCITY}).",
    )
    parser.add_argument(
        "--min-resolution",
        type=float,
        default=config.MIN_EVENT_RESOLUTION,
        help=f"Minimum resolution (Hz) for placed sound events (default: {config.MIN_EVENT_RESOLUTION}).",
    )
    parser.add_argument(
        "--max-resolution",
        type=float,
        default=config.MAX_EVENT_RESOLUTION,
        help=f"Maximum resolution (Hz) for placed sound events (default: {config.MAX_EVENT_RESOLUTION}).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=config.MIN_EVENT_DURATION,
        help=f"Minimum duration (seconds) for placed sound events (default: {config.MIN_EVENT_DURATION}).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=config.MAX_EVENT_DURATION,
        help=f"Maximum duration (seconds) for placed sound events (default: {config.MAX_EVENT_DURATION}).",
    )

    return vars(parser.parse_args())


def main(
    duration: float,
    n_static: int,
    n_moving: int,
    max_overlap: int,
    micarray: str,
    output_folder: str,
    fg_folder: str,
    bg_folder: str,
    mesh_path: str,
    ref_db: float,
    min_duration: float,
    max_duration: float,
    min_snr: int,
    max_snr: int,
    min_velocity: float,
    max_velocity: float,
    min_resolution: int,
    max_resolution: int,
) -> None:
    """
    Runs the generation using given arguments
    """

    logger.info(
        "Creating scene with the following parameters:\n"
        f"  - Duration: {duration}s\n"
        f"  - Static events: {n_static}\n"
        f"  - Moving events: {n_moving}\n"
        f"  - Max overlap: {max_overlap}\n"
        f"  - Microphone array: {micarray}\n"
        f"  - Output folder: {output_folder}\n"
        f"  - Foreground folder: {fg_folder}\n"
        f"  - Background folder: {bg_folder}\n"
        f"  - Mesh path: {mesh_path}\n"
        f"  - Reference dB: {ref_db}\n"
        f"  - SNR range: {min_snr}–{max_snr} dB\n"
        f"  - Duration range: {min_duration}–{max_duration} s\n"
        f"  - Velocity range: {min_velocity}–{max_velocity} m/s\n"
        f"  - Resolution range: {min_resolution}–{max_resolution} Hz"
    )

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    start = time()
    scene = Scene(
        duration=duration,
        backend="rlr",
        backend_kwargs=dict(
            mesh=mesh_path,
            add_to_context=False,
        ),
        sample_rate=44100,
        scene_start_dist=stats.uniform(0.0, duration - 1),
        event_start_dist=None,
        event_duration_dist=stats.uniform(min_duration, max_duration - min_duration),
        event_velocity_dist=stats.uniform(min_velocity, max_velocity - min_velocity),
        event_resolution_dist=stats.uniform(
            min_resolution, max_resolution - min_resolution
        ),
        snr_dist=stats.uniform(min_snr, max_snr - min_snr),
        fg_path=Path(fg_folder),
        bg_path=Path(bg_folder),
        max_overlap=max_overlap,
        ref_db=ref_db,
        allow_duplicate_audios=False,
    )

    scene.add_microphone(microphone_type=micarray, alias=micarray)

    for _ in range(n_static):
        scene.add_event(event_type="static")

    for _ in range(n_moving):
        scene.add_event(event_type="moving")

    scene.add_ambience()

    audio_path = str(output_path / "audio_out.wav")
    metadata_path = str(output_path / "metadata_out.json")
    scene.generate(
        audio_fname=audio_path,
        metadata_fname=metadata_path,
    )
    end = time() - start

    # Add the time taken into the metadata dictionary
    js = scene.to_dict()
    js["time"] = end

    # Dump the metadata to a JSON
    with open(metadata_path, "w") as f:
        json.dump(js, f, indent=4, ensure_ascii=False)

    logger.info(
        f"Finished rendering in {end:.4f} seconds.\n"
        f"  - The saved audio is in {audio_path}\n"
        f"  - The metadata is in {metadata_path}\n"
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
