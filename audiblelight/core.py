import os
import numpy as np
import json
import soundfile as sf

# import Space ???

class Scene:
    def __init__(
        self,
        duration,
        mesh_path,
        mic_array_name,
        fg_path,
        ref_db=-50,
        event_time_dist=('uniform', 0.0, 10.0),
        event_duration_dist=('uniform', 0.5, 4.0),
        event_velocity_dist=('uniform', 0.1, 1.5),
        snr_dist=('uniform', 6, 30),
        max_overlap=3,
        mic_array_position=None  # Optional override
    ):
        self.duration = duration
        self.mesh = load_mesh(mesh_path)
        self.fg_path = fg_path
        self.ref_db = ref_db
        self.event_time_dist = event_time_dist
        self.event_duration_dist = event_duration_dist
        self.event_velocity_dist = event_velocity_dist
        self.snr_dist = snr_dist
        self.max_overlap = max_overlap # time overlaps (we could include a space overlaps parameter too)

        self.events = []

        # define scene here?

        self.ambience_enabled = False

    def add_ambience(self):
        """Add default room ambience (e.g., Brownian noise)."""
        self.ambience_enabled = True

    def add_event(
        self,
        label=('choose', []),
        source_file=('choose', []),
        source_time=('const', 0.0),
        event_time=None,
        event_duration=None,
        event_velocity=None,
        snr=None,
    ):
        """Add a foreground event with optional per-event overrides."""
        event_time = sample_distribution(event_time or self.event_time_dist)
        event_duration = sample_distribution(event_duration or self.event_duration_dist)
        snr = sample_distribution(snr or self.snr_dist)
        # the event velocity is an interesting case, as it will need to be dependent on:
        #   whether the event is moving or not
        #   its duration
        #   note possible final location is itself a function of the above
        event_velocity = sample_distribution(event_velocity or self.event_velocity_dist)

        # Validate no excessive overlaps
        if self._would_exceed_overlap(event_time, event_duration):
            return  # Skip or retry if needed. Let's borrow from SpatialScaper how this is handled, inspiration at least

        # Sample initial position and compute trajectory
        init_pos = sample_random_position(self.mesh)
        traj = self._generate_trajectory(init_pos, event_duration, event_velocity)
        # This could be a linear trajectory, an arch or a random walk. Some of these are implemented already in SpatialScaper

        event = Event(
            label=label,
            source_file=source_file,
            source_time=source_time,
            start_time=event_time,
            duration=event_duration,
            velocity=event_velocity,
            trajectory=traj,
            snr=snr
        )
        self.events.append(event)

    def generate(self, audio_path, metadata_path, spatial_audio_format='A'):
        """Render scene to disk."""
        audio = render_scene_audio(
            mesh=self.mesh,
            mic_array=self.mic_array,
            events=self.events,
            duration=self.duration,
            ambience=self.ambience_enabled,
            ref_db=self.ref_db,
            spatial_format=spatial_audio_format
        )
        sf.write(audio_path, audio, samplerate=48000) # we shouldn't hard-code this

        metadata = {
            'duration': self.duration,
            'mesh': str(self.mesh),
            'mic_array': self.mic_array.to_dict(),
            'events': [e.to_dict() for e in self.events],
            'ref_db': self.ref_db,
            'ambience': self.ambience_enabled,
            'spatial_format': spatial_audio_format
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
