#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration file containing default values used throughout the entire package."""

# Audio
SAMPLE_RATE = 44100
BUFFER_SIZE = 8192

# Scene
SCENE_DURATION = 60
REF_DB = -65
MAX_OVERLAP = 2
WARN_WHEN_SCENE_DURATION_BELOW = 5

# Event
DEFAULT_MOVING_TRAJECTORY = "linear"
MIN_EVENT_VELOCITY, MAX_EVENT_VELOCITY = 1.0, 2.0
MIN_EVENT_RESOLUTION, MAX_EVENT_RESOLUTION = 1.0, 4.0
MIN_EVENT_DURATION, MAX_EVENT_DURATION = 2.0, 10.0
MIN_EVENT_SNR, MAX_EVENT_SNR = 5.0, 30.0
#  Define averages: these are helpful in cases where we need a "single"
#  default value, i.e. where a range or distribution isn't needed
#  We just use the midpoint between the two extremes, but this could be anything
AVG_EVENT_VELOCITY = (MAX_EVENT_VELOCITY - MIN_EVENT_VELOCITY) / 2
AVG_EVENT_RESOLUTION = (MAX_EVENT_RESOLUTION - MIN_EVENT_RESOLUTION) / 2
AVG_EVENT_DURATION = (MAX_EVENT_DURATION - MIN_EVENT_DURATION) / 2
AVG_EVENT_SNR = (MAX_EVENT_SNR - MIN_EVENT_SNR) / 2

#  Note: Scene and Event constants taken from SpatialScaper.Scaper.__init__
#  + example_generation.py files. latter takes precedence for any conflict

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
WARN_WHEN_RAY_EFFICIENCY_BELOW = 0.5
# Max number of times we'll attempt to place a source or microphone before giving up
MAX_PLACE_ATTEMPTS = 1000

# Benchmarking settings
MIN_STATIC_EVENTS, MAX_STATIC_EVENTS = 1, 10
MIN_MOVING_EVENTS, MAX_MOVING_EVENTS = 0, 6
DEFAULT_STATIC_EVENTS = 4
DEFAULT_MOVING_EVENTS = 1
MIC_ARRAY_TYPE = "ambeovr"
N_SCENES = 1000
