# Downloading datasets

This folder contains scripts for downloading the following datasets, which may prove handy when working with `AudibleLight`:

- [Gibson Database of 3D Spaces](http://gibsonenv.stanford.edu/database/): `download_gibson.py`
- [Free Music Archive](https://github.com/mdeff/fma): `download_fma.py`
- [FSD50K](https://zenodo.org/records/4060432): `download_fsd.py`

Note that, by running these scripts, you confirm that you agree to abide by their terms of use. In particular, for the *Gibson Database of 3D Spaces*, you confirm that you have signed and completed the associated [user agreement form](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform). 

## Running the scripts

First, ensure that you have installed `AudibleLight`. Then, from the command line:

```bash
make download
```

Additional arguments to the individual scripts called by this command include:
- `--path`: the path to download the data to, defaults to `root/resources/meshes` or `root/resources/soundevents` depending on the script being run.
- `--cleanup`: whether to remove additional files not needed by `AudibleLight` (e.g., `.zip`, `.navmesh` files).
- `--remote`: remote datasets to download: e.g., for `fma`, this can be `--remote fma_small`,  `--remote fma_full`, etc.
  - Provide this argument multiple times to download multiple remote datasets.

These scripts have been adapted from [`spatialscaper`](https://github.com/marl/SpatialScaper/tree/main/scripts).

