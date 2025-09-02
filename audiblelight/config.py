#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration file containing default values used throughout the entire package."""

# Audio
SAMPLE_RATE = 44100
BUFFER_SIZE = 8192

# Scene
REF_DB = -65
MAX_OVERLAP = 3
MAX_DEFAULT_DURATION = 10

# Event


# WorldState
MESH_UNITS = "meters"
MIN_AVG_RAY_LENGTH = 3.0
#  Minimum distance one emitter can be from another
EMPTY_SPACE_AROUND_EMITTER = 0.2
#  Minimum distance one emitter can be from the mic
EMPTY_SPACE_AROUND_MIC = 0.1
#  Minimum distance from the nearest mesh surface
EMPTY_SPACE_AROUND_SURFACE = 0.2
#  Minimum distance from individual microphone capsules
EMPTY_SPACE_AROUND_CAPSULE = 0.05
#  When the ray efficiency is below this value, raise a warning in .simulate
WARN_WHEN_EFFICIENCY_BELOW = 0.5
# Max number of times we'll attempt to place a source or microphone before giving up
MAX_PLACE_ATTEMPTS = 1000
