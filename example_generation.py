import audiblelight as al
import numpy as np
import os

# OUTPUT DIRECTORY
outfolder = 'spatial_scenes'

# PATHS
fg_folder = 'audio/soundbank/foreground/'
mesh_path = 'meshes/house_model.glb'  # Mesh can be a "building" and may contain multiple rooms

# SCENE SETTINGS
n_scenes = 1000
duration = 10.0  # seconds
mic_array_name = 'tetra'
min_events = 1
max_events = 9
max_overlap = 3

# SCENE-WIDE DISTRIBUTIONS
event_time_dist = ('uniform', 0.0, duration)
event_duration_dist = ('uniform', 0.5, 4.0)
event_velocity_dist = ('uniform', 0.1, 1.5)  # meters per second
snr_dist = ('uniform', 6, 30)
ref_db = -50

# Generate scenes
for i in range(n_scenes):
    print(f'Generating spatial scene: {i+1}/{n_scenes}')

    # Create a scene object with global configuration
    scene = al.Scene(
        duration=duration,
        mesh_path=mesh_path,
        mic_array_name=mic_array_name,
        fg_path=fg_folder,
        ref_db=ref_db,
        event_time_dist=event_time_dist,
        event_duration_dist=event_duration_dist,
        event_velocity_dist=event_velocity_dist,
        snr_dist=snr_dist,
        max_overlap=max_overlap
    )

    # Add ambient background noise
    scene.add_ambience()

    # Add a random number of events
    n_events = np.random.randint(min_events, max_events + 1)
    for _ in range(n_events):
        scene.add_event(
            label=('choose', []),
            source_file=('choose', []),
            source_time=('const', 0.0)
        )

    # Render to disk
    scene.generate(
        audio_path=os.path.join(outfolder, f'scene_spatial_{i:04d}.wav'),
        metadata_path=os.path.join(outfolder, f'scene_spatial_{i:04d}.json'),
        spatial_audio_format='A'  # A-format ambisonics
    )
