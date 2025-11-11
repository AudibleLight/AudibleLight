# Spatial semantic segmentation of sound scenes

This folder contains scripts for generating spatial semantic segmentation datasets using `AudibleLight`

## Running the scripts

The main script is `generate_dataset.py` First, ensure that you have installed `AudibleLight` by running `make install`. 

Then, from the command line:

```bash
poetry run python scripts/ssseg/generate_dataset.py
```

Additional arguments can be accessed for this script by passing in the `--help` flag, or by modifying the `CONFIG` dictionary inside the script.

The code in this script is broadly inspired by the [DCASE2025 task 4 baseline](https://github.com/nttcslab/dcase2025_task4_baseline/) repository.