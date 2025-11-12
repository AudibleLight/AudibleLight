# Sound Event Localization and Detection

This folder contains scripts for generating sound event localization and detection datasets using `AudibleLight`

## Running the scripts

The main script is `generate_dataset.py` First, ensure that you have installed `AudibleLight` by running `make install`. 

Then, from the command line:

```bash
poetry run python scripts/seld/generate_dataset.py
```

Additional arguments can be accessed for this script by passing in the `--help` flag.

