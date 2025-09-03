#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert a folder of AudibleLight metadata JSON files to DCASE format."""

import argparse
from pathlib import Path

from tqdm import tqdm

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.synthesize import generate_dcase2024_metadata

DEFAULT_DIR = utils.get_project_root() / "spatial_scenes"


def process_json(filepath: Path) -> None:
    """
    Load up a Scene from the json `filepath`, convert to DCASE format, and dump CSV files
    """
    sc = Scene.from_json(filepath)
    dcase_meta = generate_dcase2024_metadata(sc)
    for mic, df in dcase_meta.items():
        outp = filepath.with_suffix(".csv").with_stem(
            f"{filepath.with_suffix('').name}_{mic}"
        )
        df.to_csv(outp, sep=",", encoding="utf-8", header=None)


def main(dirpath: str):
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory {dirpath} does not exist!")

    js_files = sorted(list(dirpath.rglob("*.json")))
    if len(js_files) == 0:
        raise ValueError("No JSON files found!")

    for filepath in tqdm(js_files, desc="Converting metadata..."):
        process_json(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of AudibleLight metadata JSON files to DCASE format."
    )
    parser.add_argument("-d", "--dirpath", type=str, default=DEFAULT_DIR)
    args = vars(parser.parse_args())
    main(**args)
