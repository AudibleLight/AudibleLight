<h1 align="center">AudibleLight 🔈💡</h1>
<h2 align="center">Spatial soundscape synthesis using ray-tracing</h2>

<p align="center">
<a href="https://github.com/AudibleLight/AudibleLight/actions"><img alt="Actions Status" src="https://github.com/AudibleLight/AudibleLight/actions/workflows/tests.yml/badge.svg"></a>
<a href="https://www.linux.org"><img alt="Platform: Linux" src="https://img.shields.io/badge/Platform-linux-lightgrey?logo=linux"></a>
<a href="https://www.python.org/"><img alt="Python: 3.10" src="https://img.shields.io/badge/Python-3.10%2B-orange?logo=python"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/AudibleLight/AudibleLight/pulls"><img alt="Pull requests: welcome" src="https://img.shields.io/badge/pull_requests-welcome-blue?logo=github"></a>
</p>

> [!WARNING]
> *This project is currently under heavy development*. We have done our due diligence to ensure that it works as expected. However, if you encounter any errors, please [open an issue](https://github.com/AudibleLight/AudibleLight/issues) and let us know.

**Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [Roadmap](#roadmap)

## What is `AudibleLight`?

This project provides a platform for generating synthetic soundscapes by simulating arbitrary microphone configurations and dynamic sources in both parameterized and 3D-scanned rooms. Under the hood, `AudibleLight` uses Meta’s [open-source acoustic ray-tracing engine](https://github.com/beasteers/rlr-audio-propagation) to simulate spatial room impulse responses and convolve them with recorded events to emulate array recordings of moving sources. The resulting soundscapes can prove useful in training models for a variety of downstream tasks, including acoustic imaging, sound event localisation and detection, direction of arrival estimation, etc.

In contrast to other projects (e.g., [`sonicsim`](https://github.com/JusperLee/SonicSim/tree/main/SonicSim-SonicSet), [`spatialscaper`](https://github.com/marl/SpatialScaper)), `AudibleLight` provides a straightforward API without restricting the user to any specific dataset. You can bring your own mesh and your own audio files, and `AudibleLight` will handle all the spatial logic, validation, and synthesis necessary to ensure that the resulting soundscapes are valid for use in training machine learning models and algorithms.

`AudibleLight` is developed by researchers at the [Centre for Digital Music, Queen Mary University of London](https://www.c4dm.eecs.qmul.ac.uk/) in collaboration with [Meta Reality Labs](https://www.meta.com/en-gb/emerging-tech).

## Installation:

### Prerequisites

- `git`
- `python3.10` or above (tested up to 3.12)
- `poetry`
- A modern Linux distro: current versions of `Ubuntu` and `Red Hat` have been tested and confirmed to work.
  - Using another OS? Let us know so we can add it here!

### Install via the command line

```bash
sudo apt update
sudo apt install libsox-dev libsox-fmt-all freeglut3-dev
git clone https://github.com/AudibleLight/AudibleLight.git
poetry install
```

### Install via `pypi`

***Coming soon!***

### Running `pre-commit` hooks

```bash
poetry run pre-commit install
pre-commit run --all-files
```

## Usage

### Script

To generate a simple audio scene with a set number of moving and static sound sources, run:
```bash
poetry run python scripts/generate_with_random_events.py
```

To see the available arguments that this script takes, add the `--help` argument

### Notebook

An example notebook showing placement of static and moving sound sources can be found inside `notebooks/example_generation.py`.

## Contributions

... are welcome! Please [make a PR](https://github.com/AudibleLight/AudibleLight/pulls) or take a look [at the open issues](https://github.com/AudibleLight/AudibleLight/issues).

### Running the tests

Before making a PR, ensure that you run the following commands:

```bash
poetry run flake8 audiblelight --count --select=E9,F63,F7,F82 --show-source --statistics
poetry run pytest -n auto -vv --cov-report term-missing --cov-report=xml --cov=audiblelight tests
```

These are identical to the commands currently run by our CI pipeline.

## Roadmap

- Feature parity with `spatialscaper`
- Spatial audio augmentations (from https://arxiv.org/abs/2101.02919)
- Add to `pypi` (i.e., allowing `pip install audiblelight`)
- HRTF support
- Directional microphone capsules support
- Increased visualisation options

### API Sketch

<img width="3748" height="1454" alt="Screenshot from 2025-07-21 10-52-03" src="https://github.com/user-attachments/assets/52d3df17-126b-43c6-8e57-0a724e74e6ef" />
