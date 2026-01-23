# Downloading datasets

This folder contains scripts for downloading the following datasets, which may prove handy when working with `AudibleLight`:

- [Gibson Database of 3D Spaces](http://gibsonenv.stanford.edu/database/): `download_gibson.py`
  - Also `download_gibson_waypoints.py`, which downloads [preset navigation waypoints for the same spaces](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md#navigation-waypoints).
- [Free Music Archive](https://github.com/mdeff/fma): `download_fma.py`
- [FSD50K](https://zenodo.org/records/4060432): `download_fsd.py`
- [SpatialScaper RIRs](https://github.com/marl/SpatialScaper/tree/main?tab=readme-ov-file#preparing-rir-datasets): `download_rirs.py`
- [VisualGenome Images](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html): `download_visualgenome.py`

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

## A note about `VisualGenome`

This dataset contains an extremely large number of annotated images, only a small minority of which are likely to be useful to `AudibleLight` users.

As such, this script hardcodes a few object classes that are likely to be of most use, and only downloads these images. These classes are broadly the same as those commonly used in the [DCASE Challenge](https://dcase.community) for Sound Event Localisation and Detection.

In order to modify the image classes that are downloaded, you need to modify the `DCASE_VG_SELECTED` variable inside `download_visualgenome.py`. This variable is a dictionary, where the keys correspond with new object names (can be anything) and the values a list of VisualGenome `name` types (must be contained inside `objects.json`, downloaded from VisualGenome). For example:

```python
DCASE_VG_SELECTED = {
    "telephone": ["telephone", "phone", "cellphone"],
}
```

will download and crop all VisualGenome objects tagged with `name IN [telephone, phone, cellphone]`, and extract them to the new folder `telephone`.
