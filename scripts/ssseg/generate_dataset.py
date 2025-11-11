#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates an example dataset for SSSEG, similar to DCASE 2025 Task 4.

By default, this script can generate:
- 100000 10-second long spatial recordings
- Sampling rate of 34kHz
- First-order ambisonics audio format
- Between 1 and 3 static foreground events, with an SNR between 5 and 20
- Between 1 and 2 static "interference" events, with an SNR between 0 and 15
- Background audio randomly sampled from real-world recordings
- Maximum polyphony of 3 (with NO same class foreground audio events)

The audio outputs of the script include:
- "Dry" audio files for every foreground Event, padded to the duration of the Scene
    - These audio files represent only the direct path and early reflections of the event’s sound as captured by a
      specific reference channel in the provided impulse responses (IRs).
    - These audio files are given in MONO format
- "Wet" audio files for every foreground Event, padded to the duration of the Scene
    - These audio files represent the "spatialized" sound event, without any other foreground or interference events,
      or the background noise added to the Scene.
    - These audio files are in FOA format
- The complete soundscape, including all foreground, background, and interference events
    - These audio files are in FOA format

The metadata outputs of the script include:
- A simple JSON configuration file, showing the configuration, SOFA file, and foreground/background Events
  (including source file, timestamps, SNR, etc.)

The backend is hardcoded to SOFA (i.e., similar operation to `SpatialScaper`). Many other parameters can be controlled
using flags passed in to the script.
"""

import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
from loguru import logger
from tqdm import tqdm

from audiblelight import utils
from audiblelight.class_mappings import sanitize_class_mapping
from audiblelight.core import Scene

CONFIG = {
    "snr_range": [5, 20],
    "nevent_range": [1, 3],
    "interference_snr_range": [0, 15],
    "ninterference_range": [1, 2],
    "dataset_length": 100000,
    "shuffle_label": False,
    "foreground_dir": utils.get_project_root() / "resources/soundevents",
    "background_dir": utils.get_project_root() / "resources/noise",
    "rir_dir": utils.get_project_root() / "resources/sofa/rirs",
    "interference_dir": utils.get_project_root() / "resources/interference",
    "output_dir": utils.get_project_root() / "spatial_scenes_ssseg",
    "duration": 10.0,
    "sr": 32000,
    "max_event_overlap": 3,
    "ref_db": -50,
    "ref_ir_channel": 0,
    "direct_path_time_ms": [5, 60],
    "n_scapes": 100000,
    "return_dry": True,
    "return_wet": True,
    "label_set": "dcase2025task4",
}


# noinspection PyProtectedMember
def generate(generation_idx: int, **config) -> None:
    """
    Makes a single generation and saves outputs
    """
    # Randomly select a SOFA path
    sofa_path = random.choice(
        utils.sanitise_directory(config["rir_dir"]).rglob("*.sofa")
    )

    # Uncomment to use a hardcoded SOFA path
    # sofa_path = utils.get_project_root() / "tests/test_resources/metu_foa.sofa"

    # We want to add an extra key-value pair to our mapping so we can keep our interference audios separate
    init_mapper = sanitize_class_mapping(config["label_set"]).mapping
    class_mapping = {
        **init_mapper,
        "interference": max(init_mapper.values()) + 1,
    }

    # Create the scene with the required sofa path
    ssc = Scene(
        backend="sofa",
        duration=config["duration"],
        sample_rate=config["sr"],
        fg_path=config["foreground_dir"],
        bg_path=config["background_dir"],
        # Cap duration of Events between 0 and 5 seconds
        event_duration_dist=lambda: random.uniform(0, config["duration"] / 2),
        allow_same_class_events=False,
        allow_duplicate_audios=False,
        class_mapping=class_mapping,
        ref_db=config["ref_db"],
        max_overlap=config["max_event_overlap"],
        backend_kwargs=dict(sofa=sofa_path, mic_alias="ssseg_mic"),
    )

    # Make a random selection for the number of foreground events and add them
    nevents = random.randint(config["nevent_range"][0], config["nevent_range"][1])
    for _ in range(nevents):
        try:
            ssc.add_event(
                event_type="static",
                snr=random.uniform(config["snr_range"][0], config["snr_range"][1]),
                ref_ir_channel=config["ref_ir_channel"],
                direct_path_time_ms=config["direct_path_time_ms"],
            )
        except ValueError as e:
            logger.error(e)

    # Make a random selection for the number of interference events and add them
    ninterferences = random.randint(
        config["ninterference_range"][0], config["ninterference_range"][1]
    )
    interference_audios = ssc._introspect_audio_directories(
        [config["interference_dir"]]
    )
    for _ in range(ninterferences):
        try:
            interference_event = ssc.add_event(
                event_type="static",
                snr=random.uniform(
                    config["interference_snr_range"][0],
                    config["interference_snr_range"][1],
                ),
                filepath=ssc._get_random_audio(interference_audios),
            )
            # Manually update the label and class ID for this event
            interference_event.class_id = None
            interference_event.class_label = "interference"

        except ValueError as e:
            logger.error(e)

    # Randomly add a background audio file in
    background_audios = ssc._introspect_audio_directories([config["background_dir"]])
    ssc.add_ambience(channels=4, filepath=random.choice(background_audios))

    # Do the generation but don't save any outputs just yet
    ssc.generate(audio=False, metadata_json=False, metadata_dcase=False)

    # Grab all non-interference Events
    foreground_events = [
        v_ for v_ in ssc.get_events() if v_.class_label != "interference"
    ]

    # Sanity check that all Events have unique IDs
    all_ids = [fg.class_id for fg in foreground_events]
    assert len(set(all_ids)) == len(all_ids), "Duplicated sound events in the mixture"

    # Zero pad the generation IDX for nice sorting of saved files
    generation_idx = str(generation_idx).zfill(6)
    outdir = config["output_dir"]

    # Iterate over all foreground events
    for fg_idx, fg in enumerate(foreground_events):
        # Grab wet and dry audio
        audio_dry = fg._spatial_audio_dry_padded["ssseg_mic"]
        audio_wet = fg._spatial_audio_padded["ssseg_mic"]

        # Sanity checking audio file shapes
        assert (
            audio_dry.shape[-1]
            == audio_wet.shape[-1]
            == (ssc.sample_rate * ssc.duration)
        )
        assert audio_wet.shape[0] == 4

        # Create the filepaths
        outpath_dry = (
            outdir / f"dry/scape{generation_idx}_event{str(fg_idx).zfill(3)}.wav"
        )
        outpath_wet = (
            outdir / f"wet/scape{generation_idx}_event{str(fg_idx).zfill(3)}.wav"
        )

        # Dump the audio to disk
        if config["return_dry"]:
            sf.write(outpath_dry, audio_dry, ssc.sample_rate)
        if config["return_wet"]:
            sf.write(outpath_wet, audio_wet.T, ssc.sample_rate)

    # Dump the scape audio to disk, too
    audio_scape = ssc.audio["ssseg_mic"]
    outpath_scape = outdir / f"scape/scape{generation_idx}.wav"
    sf.write(outpath_scape, audio_scape.T, ssc.sample_rate)

    # Finally, dump the metadata using the SpatialScaper format
    outpath_metadata = outdir / f"metadata/scape{generation_idx}.json"
    metadata = {
        "config": {
            k_: v_ if not isinstance(v_, Path) else str(v_.resolve())
            for k_, v_ in config.items()
        },
        "fg_events": [
            {
                "label": c.class_label,
                "source_file": str(c.filepath.resolve()),
                "source_time": c.scene_start,
                "event_time": c.event_start,
                "event_duration": c.duration,
                "snr": c.snr,
                "role": "foreground",
                "pitch_shift": None,
                "time_stretch": None,
                "event_position": utils.coerce_nested_inputs(
                    c.start_coordinates_absolute
                ),
            }
            for c in foreground_events
        ],
        "bg_events": [
            {
                "label": None,
                "source_file": str(c.filepath.resolve()),
                "source_time": 0,
                "event_time": 0,
                "event_duration": ssc.duration,
                "snr": 0,
                "role": "background",
                "pitch_shift": None,
                "time_stretch": None,
                "event_position": None,
            }
            for c in ssc.get_ambiences()
        ],
        "room": str(sofa_path.resolve()),
    }
    with open(outpath_metadata, "w") as outfile:
        json.dump(metadata, outfile, indent=4)


def main(**config):
    # Create the output folders if they don't currently exist
    outdir = Path(config["output_dir"])
    for fp in [
        outdir / "metadata",
        outdir / "dry",
        outdir / "wet",
        outdir / "scape",
    ]:
        if not fp.exists():
            os.makedirs(fp)

    for generation_idx in tqdm(range(config["n_scapes"]), desc="Generating scenes..."):
        generate(generation_idx, **config)


if __name__ == "__main__":
    # Use module docstring for the help text
    parser = ArgumentParser(description=__doc__)

    # Add all config parameters as arguments to the CLI, with defaults
    for k, v in CONFIG.items():
        parser.add_argument(f"--{k}", default=v)

    kwargs = vars(parser.parse_args())
    main(**kwargs)
