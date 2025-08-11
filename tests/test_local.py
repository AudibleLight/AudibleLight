#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define test cases that should not run on the remote server"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import soundfile as sf
from pyroomacoustics.doa.music import MUSIC
from scipy.signal import stft
from trimesh import Trimesh

from audiblelight import utils
from audiblelight.core import Scene
from audiblelight.micarrays import AmbeoVR
from audiblelight.worldstate import WorldState, load_mesh, repair_mesh
from tests import utils_tests


@pytest.mark.parametrize("mesh_fpath", utils_tests.TEST_MESHES)
@pytest.mark.parametrize("mic_type", ["ambeovr", "eigenmike32"])
@pytest.mark.parametrize("n_events, duration, max_overlap", [(1, 30, 3), (9, 50, 6)])
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_pipeline(mesh_fpath, n_events, duration, max_overlap, mic_type):
    """
    This function tests the whole pipeline, from generating a Scene with a given mesh, adding events, and creating audio
    """
    # Create the scene
    sc = Scene(
        duration=duration,
        mesh_path=mesh_fpath,
        # Use default distributions for everything
        fg_path=utils_tests.SOUNDEVENT_DIR,
        max_overlap=max_overlap,
    )
    # Add the desired microphone type and number of events
    sc.add_microphone(microphone_type=mic_type)
    for i in range(n_events):
        sc.add_event(event_type="static", emitter_kwargs=dict(keep_existing=True))
    # Generate everything and check the files exist
    sc.generate(audio_path="audio_out.wav", metadata_path="metadata_out.json")
    for path in ["audio_out.wav", "metadata_out.json"]:
        assert os.path.isfile(path)
        os.remove(path)


@pytest.mark.parametrize("mesh_fpath", utils_tests.TEST_MESHES)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_repair_mesh(mesh_fpath: str):
    # Load up the mesh
    loaded = load_mesh(mesh_fpath)
    # Make a copy of the mesh
    new_mesh = Trimesh(vertices=loaded.vertices.copy(), faces=loaded.faces.copy())
    # Repair the mesh, in-place
    repair_mesh(new_mesh)
    # Should still expect a mesh object to be returned
    assert isinstance(new_mesh, Trimesh)


# Goes (1 mic, 4 emitters), (2 mics, 3 emitters), (3 mics, 2 emitters), (4 mics, 1 emitter)
@pytest.mark.parametrize(
    "n_mics,n_emitters", [(m, s) for m, s in zip(list(range(1, 5))[::-1], range(1, 5))]
)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_simulated_ir(n_mics: int, n_emitters: int, oyens_space: WorldState):
    # For reproducible results
    utils.seed_everything(n_emitters)
    # Add some emitters and microphones to the space
    #  We could use other microphone types, but they're slow to simulate
    oyens_space.add_microphones(
        microphone_types=["ambeovr" for _ in range(n_mics)], keep_existing=False
    )
    oyens_space.add_emitters(n_emitters=n_emitters, polar=False)
    # Grab the IRs: we should have one array for every microphone
    oyens_space.simulate()
    assert isinstance(oyens_space.irs, dict)
    simulated_irs = list(oyens_space.irs.values())
    assert len(simulated_irs) == n_mics
    # Iterate over each individual microphone
    total_capsules = 0
    for mic in oyens_space.microphones.values():
        # Grab the shape of the IRs for this microphone
        actual_capsules, actual_emitters, actual_samples = mic.irs.shape
        # We should have the expected number of emitters, capsules, and samples
        assert actual_emitters == n_emitters
        assert actual_capsules == mic.n_capsules
        assert actual_samples >= 1  # difficult to test number of samples
        total_capsules += actual_capsules
    # IRs for all microphones should have same number of emitters and samples
    _, mic_1_emitters, mic_1_samples = oyens_space.get_microphone("mic000").irs.shape
    assert all(
        [m.irs.shape[1] == mic_1_emitters for m in oyens_space.microphones.values()]
    )
    assert all(
        [m.irs.shape[2] == mic_1_samples for m in oyens_space.microphones.values()]
    )
    # Number of capsules should be the same as the "raw" results of the raytracing engine
    assert total_capsules == oyens_space.ctx.get_audio().shape[0]


@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_save_wavs(oyens_space: WorldState):
    # Add some microphones and emitters
    oyens_space.add_microphone(
        microphone_type="ambeovr", keep_existing=False
    )  # just adds an ambeovr mic in a random plcae
    oyens_space.add_emitter(polar=False)
    # Run the simulation
    oyens_space.simulate()
    # Dump the IRs to a temporary directory
    with TemporaryDirectory() as tmp:
        oyens_space.save_irs_to_wav(tmp)
        # We have 1 microphone with 4 capsules and 1 sound emitter
        #  We should have saved a WAV file for each of these
        for caps_idx in range(4):
            # The WAV file should exist
            fp = os.path.join(tmp, f"mic000_capsule00{caps_idx}_emitter000.wav")
            assert os.path.exists(fp)
            # Load up the WAV file in librosa and get the number of samples
            y, _ = sf.read(
                fp,
            )
            # Compare to the original IR
            x = oyens_space.irs["mic000"][caps_idx][0]
            assert np.allclose(y, x, atol=1e-4)
    # Temporary directory is implicitly cleaned up


@pytest.mark.parametrize(
    "microphone,emitters,actual_doa",
    [
        # Test case 1: two emitters at 90 and 270 degree angles from the mic
        (
            [-1.5, -1.5, 0.7],  # mic placed in bedroom 1
            [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
            [90, 270],
        ),
        # Test case 2: two emitters at 0 and 180 degree angles from the mic
        (
            [2.9, -7.0, 0.3],  # mic placed in bedroom 2
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [0, 180],
        ),
        # Test case 3: single sound emitter at a 45-degree angle
        (
            [2.5, -1.0, 0.5],  # mic placed in living room
            [[1.0, 1.0, 0.0]],
            [
                45,
            ],
        ),
    ],
)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_simulated_doa_with_music(
    microphone: list, emitters: list, actual_doa: list[int], oyens_space: WorldState
):
    """
    Tests DOA of simulated sound emitters and microphones with MUSIC algorithm.

    Places an Eigenmike32, simulates sound emitters, runs MUSIC, checks that estimated DOA is near to actual DOA
    """
    # Add the microphones and simulate the space
    oyens_space.add_microphone(
        microphone_type="eigenmike32",
        position=microphone,
        keep_existing=False,
        alias="tester",
    )
    oyens_space.add_emitters(
        positions=emitters, mics="tester", keep_existing=False, polar=False
    )
    oyens_space.simulate()
    # TODO: in the future we should use simulated sound emitters, not the IRs
    output = oyens_space.irs

    # Create the MUSIC object
    l_: np.ndarray = oyens_space.get_microphone(
        "tester"
    ).coordinates_absolute.T  # coordinates of our capsules for the eigenmike
    fs = int(oyens_space.ctx.config.sample_rate)
    nfft = 1024
    num_emitters = oyens_space.num_emitters  # number of sound emitters we've added
    assert num_emitters == len(actual_doa) == len(emitters)  # sanity check everything
    music = MUSIC(
        L=l_,
        fs=fs,
        nfft=nfft,
        azimuth=np.deg2rad(np.arange(360)),
        num_sources=num_emitters,
    )

    # Iterating over all of our sound emitters
    for doa_deg_true, emitter_idx in zip(actual_doa, range(num_emitters)):
        # Get the IRs for this emitter: shape (N_capsules=32, 1=mono, N_samples)
        signals = np.vstack([m[:, emitter_idx, :] for m in output.values()])
        # Iterate over each individual IR (one per capsule: shape = 1, N_samples) and compute the STFT
        #  Stacked shape is (N_capsules, (N_fft / 2) + 1, N_frames)
        stft_signals = np.stack(
            [
                stft(cs, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
                for cs in signals
            ]
        )
        # Sanity check the returned shape
        x, y, _ = stft_signals.shape
        assert x == oyens_space.get_microphone("tester").n_capsules
        assert y == (nfft / 2) + 1
        # Run the music algorithm and get the predicted DOA
        music.locate_sources(stft_signals)
        doa_deg_pred = np.rad2deg(music.azimuth_recon[0])
        # Check that the predicted DOA is within a window of tolerance
        diff = abs(doa_deg_pred - doa_deg_true) % 360
        diff = min(diff, 360 - diff)  # smallest distance between angles
        assert diff <= 30


@pytest.mark.parametrize(
    "closemic_position,farmic_position,emitter_position",
    [
        # Testing "length-wise" in the room
        (
            [1.0, -9.5, 0.7],
            [1.0, 0.5, 0.7],
            [0.0, 0.5, 0.0],
        ),
        # Testing "width-wise" in the room
        (
            [0.5, -3.5, 0.7],
            [5.5, -3.5, 0.7],
            [0.5, 0.0, 0.0],
        ),
        # Testing "vertical-wise" in the room
        (
            [0.5, -3.5, 0.3],
            [0.5, -3.5, 0.9],
            [0.5, 0.0, 0.3],
        ),
    ],
)
@pytest.mark.skipif(os.getenv("REMOTE") == "true", reason="running on GH actions")
def test_simulated_sound_distance(
    closemic_position: list, farmic_position: list, emitter_position: list, oyens_space
):
    """
    Tests distance of simulated sound emitters and microphones.

    Places a emitter and two AmbeoVR microphones near and far, then checks that the sound hits the close mic before far
    """

    oyens_space.clear_microphones()
    oyens_space.clear_emitters()
    # Add the microphones and simulate the space
    oyens_space.add_microphones(
        microphone_types=["ambeovr", AmbeoVR],
        positions=[closemic_position, farmic_position],
        aliases=["closemic", "farmic"],
        keep_existing=False,
    )
    oyens_space.add_emitter(
        emitter_position, mic="closemic", keep_existing=False, polar=False
    )
    oyens_space.simulate()
    irs = oyens_space.irs
    # Shape of the IRs should be as expected
    assert len(irs) == 2
    # Get the IDX of the sample at which the sound hits both microphones
    arrival_close = min(np.flatnonzero(irs["closemic"]))
    arrival_far = min(np.flatnonzero(irs["farmic"]))
    # Should hit the closer mic before the further mic
    assert arrival_close < arrival_far
