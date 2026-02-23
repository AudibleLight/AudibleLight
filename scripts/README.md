# AudibleLight Scripts

This directory contains scripts that can be useful for various tasks. Alternatively, you can modify them to create your own data generation pipelines!

## `imaging`

If you're looking to generate synthetic data for [DCASE26 Task 3](https://dcase.community/challenge2026/#semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes), this is the script to use!

To run, simply use:

```
cd audiblelight
poetry run python scripts/imaging/generate_acoustic_images.py 
```

By default, the script will create the following:
- Synthetic audio files using a virtual 32-channel Eigenmike (Ambisonics A-Format, i.e. `mic`)
- Synthetic video files, with the same positioning as the Eigenmike
- Metadata .csv files, showing Event position and class (following the same format as the STARSS23 dataset: [see this page](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#dataset))
- Acoustic image JSON files, standardised following the DCASE26 dataset

## `download_data`

This directory contains scripts for downloading various asset and resource files: audio files, 3D meshes, etc. 
