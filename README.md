# AudibleLight

A library for soundscape synthesis using spatial impulse responses derived from ray-traced room scans.

This project is under heavy development, enquires should be directed to either Huw Cheston or Iran Roman (`firstinitial-surname AT qmul.ac.uk`).

## Motivation

This project provides a platform for generating synthetic soundscapes by simulating arbitrary microphone configurations and dynamic sources in both parameterized and 3D-scanned rooms. We use Meta’s [open-source acoustic ray-tracing engine](https://github.com/beasteers/rlr-audio-propagation) to simulate spatial room impulse responses and convolve them with recorded events to emulate array recordings of moving sources. The resulting soundscapes can prove useful in training models for a variety of downstream tasks, including acoustic imaging, sound event localisation and detection, direction of arrival estimation, etc.

## Installation:

- Ensure `sox` and `ffmpeg` are installed, i.e. `sudo apt install libsox-dev` etc.
- Ensure `glut` is installed, i.e. `sudo apt install freeglut3-dev`
- `git clone` this repository
- Create a `venv` and activate
- `pip install -r requirements.txt`
- Install pre-commit hooks: `pre-commit install`
- Run pre-commit: `pre-commit run --all-files`

## API Sketch

<img width="3748" height="1454" alt="Screenshot from 2025-07-21 10-52-03" src="https://github.com/user-attachments/assets/52d3df17-126b-43c6-8e57-0a724e74e6ef" />
