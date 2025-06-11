#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, variables, objects etc."""

import os
from pathlib import Path

import random
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOREGROUND_DIR = "FSD50K_FMA_SMALL"
RIR_DIR = None
FORMAT = 'foa'    # First-order ambisonics by default
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "output"  # Directory to store the generated soundscapes
REF_DB = -65  # Reference decibel level for the background ambient noise. Try making this random too!
NSCAPES = 20
SEED = 42
SAMPLE_RATE = 44100    # Default to 44.1kHz sample rate


def set_seed(seed: int = SEED) -> None:
    """Set the random seeds for libraries used by AudibleLight."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)


def get_project_root() -> str:
    """Returns the root directory of the project"""
    # Possibly the root directory, but doesn't work when running from the CLI for some reason
    poss_path = str(Path(__file__).parent.parent)
    # The root directory should always have these files (this is pretty hacky)
    if all(fp in os.listdir(poss_path) for fp in ["audiblelight", "notebooks", "resources", "tests", "setup.py"]):
        return poss_path
    else:
        return os.path.abspath(os.curdir)
