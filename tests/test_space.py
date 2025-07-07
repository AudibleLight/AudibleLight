#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/space.py"""

import os
from tempfile import TemporaryDirectory

import pytest
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from trimesh import Trimesh, Scene
from scipy.signal import stft
from pyroomacoustics.doa.music import MUSIC

from audiblelight import utils
from audiblelight.space import Space, load_mesh, repair_mesh
from audiblelight.micarrays import MICARRAY_LIST, AmbeoVR


TEST_RESOURCES = utils.get_project_root() / "tests/test_resources"
TEST_MESHES = [TEST_RESOURCES / glb for glb in TEST_RESOURCES.glob("*.glb")]


@pytest.mark.parametrize("mesh_fpath", TEST_MESHES)
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


@pytest.mark.parametrize("mesh_fpath", TEST_MESHES)
def test_load_mesh_from_fpath(mesh_fpath: str):
    loaded = load_mesh(mesh_fpath)
    assert isinstance(loaded, Trimesh)
    assert loaded.metadata["fpath"] == str(mesh_fpath)    # need both to be a string, or we'll get TypeError
    assert loaded.units == utils.MESH_UNITS    # units should be in meters
    # If we try to load from a mesh object, should raise an error
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = load_mesh(loaded)


@pytest.mark.parametrize("mesh_fpath,expected", [("iamnotafile", FileNotFoundError), (1234, TypeError)])
def test_load_broken_mesh(mesh_fpath: str, expected):
    with pytest.raises(expected):
        load_mesh(mesh_fpath)


@pytest.fixture(scope="function")
def oyens_space() -> Space:
    """Returns a Space object with the Oyens mesh (Gibson)"""
    oyens = os.path.join(utils.get_project_root(), "tests/test_resources/Oyens.glb")
    space = Space(
        oyens,
        min_distance_from_source=0.2,    # all in meters
        min_distance_from_mic=0.1,    # all in meters
        min_distance_from_surface=0.2    # all in meters
    )
    return space


@pytest.mark.parametrize(
    "microphone_type,position,alias",
    [
        (None, None, None),    # places 1 mono mic in a random position with default alias
        (None, [-0.5, -0.5, 0.5], None),  # places mono mic in assigned position
        ("eigenmike32", None, "customalias000"),     # places eigenmike32 in random position
        (AmbeoVR, None, "ambeovr000"),  # same but with AmbeoVR
        ("eigenmike32", [-0.5, -0.5, 0.5], "customeigenmike000"),    # places eigenmike32 in assigned position
        ("ambeovr", [-0.1, -0.1, 0.6], "ambeovr000"),    # places AmbeoVR in assigned position
    ]
)
def test_add_microphone(microphone_type, position, alias, oyens_space: Space):
    """Test adding a single microphone to the space"""
    # Add the microphones to the space: keep_existing=False ensures we remove previously-added microphones
    oyens_space.add_microphone(microphone_type, position, alias, keep_existing=False)
    assert isinstance(oyens_space.microphones, dict)
    assert len(oyens_space.microphones) == 1    # only addding one microphone with this function
    if alias is not None:
        assert list(oyens_space.microphones.keys())[0] == alias     # alias should be what we expect
        mic = oyens_space.microphones[alias]
    else:
        mic = oyens_space.microphones["mic000"]
    # Should have exactly 1 listener for every microphone capsule
    n_capsules = sum([m.n_capsules for m in oyens_space.microphones.values()])
    assert n_capsules == oyens_space.ctx.get_listener_count()
    valid_mics = [ma().name for ma in MICARRAY_LIST]
    # Microphone array type should be valid
    assert mic.name in valid_mics
    # Microphones should have coordinates assigned to them
    assert mic.coordinates_absolute is not None
    assert mic.coordinates_center is not None
    # Iterate over all capsules
    for capsule in mic.coordinates_absolute:
        # TODO: maybe flaky?
        assert oyens_space._is_point_inside_mesh(capsule)


# noinspection PyTypeChecker
def test_place_invalid_microphones(oyens_space: Space):
    # Trying to access IRs before placing anything should raise an error
    with pytest.raises(AttributeError):
        _ = oyens_space.irs
    # Cannot add sources with invalid input types
    for inp in [-1, [], {}, object, set(), lambda x: x]:
        with pytest.raises(TypeError):
            oyens_space.add_microphone(microphone_type=inp, keep_existing=False)
    # Cannot add mic that is way outside the mesh
    for inp in [[1000., 1000., 1000.], [-1000, -1000, -1000]]:
        with pytest.raises(ValueError):
            oyens_space.add_microphone(position=inp, keep_existing=False)
    # Cannot add alias that is already in the dictionary
    oyens_space.add_microphone(alias="tmp_alias")
    with pytest.raises(KeyError):
        oyens_space.add_microphone(alias="tmp_alias", keep_existing=True)


@pytest.mark.parametrize(
    "microphone_types,positions,aliases",
    [
        # Two eigenmikes with default aliases
        (["eigenmike32", "eigenmike32"], np.array([[-0.5, -0.5, 0.5], [0.6, 0.4, 0.4] ]), None),
        # Eigenmike32 and ambeovr in assigned positions with assigned aliases
        (["eigenmike32", "ambeovr"], [[-0.5, -0.5, 0.5], [-0.1, -0.1, 0.6]], ["eigenmike000", "ambeovr000"]),
        # Three ambeovrs in random positions with default aliases
        (["ambeovr", "ambeovr", AmbeoVR], None, None),
    ]
)
def test_add_microphones(microphone_types, positions, aliases, oyens_space: Space):
    # Add microphone types into space
    oyens_space.add_microphones(microphone_types, positions, aliases, keep_existing=False, raise_on_error=True)
    # Check overall dictionary
    assert len(oyens_space.microphones) == len(microphone_types)
    assert isinstance(oyens_space.microphones, dict)
    # Should have exactly 1 listener for every microphone capsule
    n_capsules = sum([m.n_capsules for m in oyens_space.microphones.values()])
    assert n_capsules == oyens_space.ctx.get_listener_count()
    # Iterate over all microphones
    valid_mics = [ma().name for ma in MICARRAY_LIST]
    for mic in oyens_space.microphones.values():
        # Microphone array type should be valid
        assert mic.name in valid_mics
        # Microphones should have coordinates assigned to them
        assert mic.coordinates_absolute is not None
        assert mic.coordinates_center is not None
        # Iterate over all capsules
        for capsule in mic.coordinates_absolute:
            assert oyens_space._is_point_inside_mesh(capsule)
    # If we've provided aliases, check these are correct
    if aliases is not None:
        assert all(a in oyens_space.microphones.keys() for a in aliases)
    # If we've provided positions, check these are correct
    if positions is not None:
        for expected_pos, actual_mic in zip(positions, oyens_space.microphones.values()):
            assert np.array_equal(expected_pos, actual_mic.coordinates_center)


# noinspection PyTypeChecker
def test_add_microphones_invalid_inputs(oyens_space: Space):
    # Trying to add non-unique aliases raises an error
    with pytest.raises(ValueError):
        oyens_space.add_microphones(None, None, ["ambeovr", "ambeovr"], keep_existing=False)
    # Trying to add iterables with different lengths raises an error
    with pytest.raises(ValueError):
        oyens_space.add_microphones(["ambeovr", "ambeovr"], None, ["ambeovr"], keep_existing=False)
    # Trying to add microphone outside the mesh
    for pos in [[-1000, -1000, -1000], [1000, 1000, 1000]]:
        with pytest.raises(ValueError):
            oyens_space.add_microphones(["ambeovr"], [pos], ["ambeovr"], keep_existing=False, raise_on_error=True)
    # Cannot add alias that is already in the dictionary
    oyens_space.add_microphones(aliases=["tmp_alias"])
    with pytest.raises(KeyError):
        oyens_space.add_microphones(aliases=["tmp_alias"], keep_existing=True)


@pytest.mark.parametrize(
    "test_position,expected",
    [
        (np.array([-0.4, -0.5, 0.5]), False),    # Too close to mic
        (np.array([-0.5, -0.4, 0.5]), False),    # Too close to mic
        (np.array([-0.5, -0.5, 0.4]), False),    # Too close to mic
        (np.array([-0.8, -1.5, 0.2]), False),    # Too close to the surface
        (np.array([-0.1, -0.1, 0.6]), True),    # Fine!
        (np.array([0.5, 0.5, 0.5]), True)    # Also fine
    ]
)
def test_validate_positions(test_position: np.ndarray, expected: bool, oyens_space: Space):
    """Given a microphone with coordinates [-0.5, -0.5, 0.5], test whether test_position is valid"""
    oyens_space.add_microphone(microphone_type="ambeovr", position=[-0.5, -0.5, 0.5], keep_existing=False)
    assert oyens_space._validate_position(test_position) == expected


@pytest.mark.parametrize(
    "position,source_alias",
    [
        (None, None),    # Add random source with no aliases
        ([-0.1, -0.1, 0.6], "custom_alias"),    # add specific source with custom alias
        ([-0.5, -0.5, 0.5], "custom_alias"),
    ]
)
def test_add_source(position, source_alias, oyens_space: Space):
    oyens_space._clear_microphones()
    # Add the sources in and check that the shape of the resulting array is what we expect
    oyens_space.add_source(position, source_alias, mic_alias=None, keep_existing=False)
    assert isinstance(oyens_space.sources, dict)
    assert len(oyens_space.sources) == 1
    # Get the desired source
    src = oyens_space.sources[source_alias] if source_alias is not None else oyens_space.sources["src000"]
    assert len(src) == 3
    if position is not None:
        assert np.array_equal(src, position)


def test_add_source_invalid(oyens_space: Space):
    # Raise error when no microphone with alias has been added
    with pytest.raises(KeyError):
        oyens_space.add_source(mic_alias="ambeovr", position=[1000, 1000, 1000], keep_existing=False)
    # Raise error when trying to add source out of bounds
    oyens_space.add_microphone(alias="ambeovr")
    with pytest.raises(ValueError):
        oyens_space.add_source(mic_alias="ambeovr", position=[1000, 1000, 1000], keep_existing=False)
    # Cannot add source that directly intersects with a microphone
    oyens_space.add_microphone(position=[-0.5, -0.5, 0.5], keep_existing=False)
    with pytest.raises(ValueError):
        oyens_space.add_source([-0.5, -0.5, 0.5])    # same, in absolute terms
    with pytest.raises(ValueError):
        oyens_space.add_source([0.0, 0.0, 0.0], mic_alias="mic000")    # same, in relative terms


@pytest.mark.parametrize(
    "position,accept",
    [
        ([0.1, 0.0, 0.0], False),
        (np.array([0.0, 0.1, 0.0]), False),
        ([1000, 1000, 1000], False),
        ([-0.2, 0.2, 0.2], True),
        ([0.2, -0.3, -0.2], True),
    ]
)
def test_add_source_relative_to_mic(position, accept: bool, oyens_space: Space):
    # Add a microphone to the space
    oyens_space.add_microphone(
        microphone_type="ambeovr",
        position=[-0.5, -0.5, 0.5],
        alias="tester",
        keep_existing=False
    )
    # Trying to add a source that should be rejected
    if not accept:
        with pytest.raises(ValueError):
            oyens_space.add_source(position=position, mic_alias="tester", keep_existing=False)
    else:
        oyens_space.add_source(position=position, mic_alias="tester", keep_existing=False)
        assert len(oyens_space.sources) == 1
        src = oyens_space.sources["src000"]
        assert len(src) == 3


@pytest.mark.parametrize(
    "positions,source_aliases",
    [
        (np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]), None),
        (np.array([[0.5, 0.5, 0.5], [0.6, 0.2, 0.5]]), ["custom_alias1", "custom_alias2"]),
        ([[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5], [-0.4, -0.5, 0.5]], ["custom_alias1", "custom_alias2", "custom_alias3"]),
    ]
)
def test_add_sources(positions, source_aliases, oyens_space: Space):
    oyens_space._clear_microphones()
    oyens_space.add_sources(positions, source_aliases, keep_existing=False)
    assert len(oyens_space.sources) == len(positions)
    if source_aliases is not None:
        assert set(oyens_space.sources.keys()) == set(source_aliases)
    for source in oyens_space.sources.values():
        assert oyens_space._is_point_inside_mesh(source)


@pytest.mark.parametrize(
    "test_position,expected_shape",
    [
        (np.array([[0.1, 0.0, 0.0], [-0.2, 0.2, 0.2]]), 1),    # 1: too close to mic, so skipped, 2: fine
        ([[-0.2, 0.2, 0.2], [-0.2, 0.3, 0.2]], 1),    # 1: fine, 2: too close to source 1, so skipped
        (np.array([[-0.2, 0.2, 0.2], [0.2, -0.3, -0.2]]), 2),  # both fine
    ]
)
def test_add_sources_relative_to_mic(test_position: np.ndarray, expected_shape: tuple[int], oyens_space: Space):
    # Clear everything out
    oyens_space._clear_microphones()
    oyens_space._clear_sources()
    oyens_space.add_microphone(microphone_type=AmbeoVR, position=[-0.5, -0.5, 0.5], alias="testmic", keep_existing=False)
    # Add the sources in and check that the shape of the resulting array is what we expect
    #  We set `raise_on_error=False` so we skip over raising an error for invalid sources
    oyens_space.add_sources(positions=test_position, mic_aliases="testmic", keep_existing=False, raise_on_error=False)
    assert len(oyens_space.sources) == expected_shape
    for source in oyens_space.sources.values():
        assert oyens_space._is_point_inside_mesh(source)


@pytest.mark.parametrize(
    "n_sources",
    [i for i in range(1, 10, 2)],
)
def test_add_n_sources(n_sources, oyens_space: Space):
    oyens_space.add_sources(n_sources=n_sources, keep_existing=False)
    assert len(oyens_space.sources) == n_sources
    for source in oyens_space.sources.values():
        assert oyens_space._is_point_inside_mesh(source)


@pytest.mark.parametrize(
    "test_position,expected_shape",
    [
        (np.array([[-0.4, -0.5, 0.5], [-0.1, -0.1, 0.6]]), 1),    # 1: too close to mic, 2: fine
        (np.array([[0.5, 0.5, 0.5], [0.6, 0.4, 0.5]]), 1),    # 1: fine, 2: too close to source 1
        ([[-0.1, -0.1, 0.6]], 1),
        ([[-0.1, -0.1, 0.6], [0.5, 0.5, 0.5]], 2),
    ]
)
def test_add_sources_at_specific_position(test_position: np.ndarray, expected_shape: tuple[int], oyens_space: Space):
    oyens_space.add_microphone(microphone_type=AmbeoVR, position=[-0.5, -0.5, 0.5], keep_existing=False)
    # Add the sources in and check that the shape of the resulting array is what we expect
    oyens_space.add_sources(positions=test_position, keep_existing=False, raise_on_error=False)
    assert len(oyens_space.sources) == expected_shape


def test_add_sources_invalid(oyens_space: Space):
    # n_sources must be a postive integer
    with pytest.raises(AssertionError):
        for inp in ["asdf", [], -1, -100, 0]:
            oyens_space.add_sources(n_sources=inp, keep_existing=False)
    # Cannot specify both a number of random sources and positions for them
    with pytest.raises(TypeError):
        oyens_space.add_sources(positions=[[0, 0, 0]], n_sources=1)
    # Aliases for sources must be unique
    with pytest.raises(ValueError):
        oyens_space.add_sources(source_aliases=["asdf", "asdf"])
    # Cannot add sources that are way outside the mesh
    with pytest.raises(ValueError):
        oyens_space.add_sources([[1000., 1000., 1000.], [-1000, -1000, -1000]], keep_existing=False)


@pytest.mark.parametrize("num_rays", [1, 10, 100])
def test_calculate_weighted_average_ray_length(num_rays: int, oyens_space: Space):
    # Get a random valid point inside the mesh
    point = oyens_space.get_random_position()
    result = oyens_space.calculate_weighted_average_ray_length(point, num_rays=num_rays)
    # Validate output is positive float and finite (since rays should hit mesh)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result > 0


@pytest.mark.parametrize("test_num", range(1, 5))
def test_get_random_position(test_num: int, oyens_space: Space):
    # For reproducible results
    utils.seed_everything(test_num)
    # Add some microphones and sources to the space
    for idx_ in range(test_num):
        oyens_space.add_microphone(keep_existing=True)
        oyens_space.add_source(keep_existing=True)
    # Grab a random position
    random_point = oyens_space.get_random_position()
    # It should be valid (suitable distance from surfaces, inside mesh, away from mics/sources...)
    assert oyens_space._validate_position(random_point)
    assert random_point.shape == (3,)   # should be a 1D array of XYZ


# Goes (1 mic, 4 sources), (2 mics, 3 sources), (3 mics, 2 sources), (4 mics, 1 source)
@pytest.mark.parametrize("n_mics,n_sources", [(m, s) for m, s in zip(list(range(1, 5))[::-1], range(1, 5))])
def test_simulated_ir(n_mics: int, n_sources: int, oyens_space: Space):
    # For reproducible results
    utils.seed_everything(n_sources)
    # Add some sources and microphones to the space
    #  We could use other microphone types, but they're slow to simulate
    oyens_space.add_microphones(microphone_types=["ambeovr" for _ in range(n_mics)], keep_existing=False)
    oyens_space.add_sources(n_sources=n_sources)
    # Grab the IRs: we should have one array for every microphone
    oyens_space.simulate()
    assert isinstance(oyens_space.irs, dict)
    simulated_irs = list(oyens_space.irs.values())
    assert len(simulated_irs) == n_mics
    # Iterate over each individual microphone
    total_capsules = 0
    for mic in oyens_space.microphones.values():
        # Grab the shape of the IRs for this microphone
        actual_capsules, actual_sources, actual_samples = mic.irs.shape
        # We should have the expected number of sources, capsules, and samples
        assert actual_sources == n_sources
        assert actual_capsules == mic.n_capsules
        assert actual_samples >= 1    # difficult to test number of samples
        total_capsules += actual_capsules
    # IRs for all microphones should have same number of sources and samples
    _, mic_1_sources, mic_1_samples = oyens_space.microphones["mic000"].irs.shape
    assert all([m.irs.shape[1] == mic_1_sources for m in oyens_space.microphones.values()])
    assert all([m.irs.shape[2] == mic_1_samples for m in oyens_space.microphones.values()])
    # Number of capsules should be the same as the "raw" results of the raytracing engine
    assert total_capsules == oyens_space.ctx.get_audio().shape[0]


def test_create_plot(oyens_space):
    # Add some microphones and sources
    oyens_space.add_microphone(keep_existing=False)
    oyens_space.add_source()
    # Create the plot
    fig = oyens_space.create_plot()
    assert isinstance(fig, plt.Figure)
    # Should have two axes for the two views
    assert len(fig.get_axes()) == 2


def test_create_scene(oyens_space):
    # Add some microphones and sources
    oyens_space.add_microphone(keep_existing=False)
    oyens_space.add_source()
    # Create the scene
    scene = oyens_space.create_scene()
    assert isinstance(scene, Scene)
    # Should have more geometry than the "raw" scene (without adding spheres for capsules/sources)
    assert len(scene.geometry) > len(oyens_space.mesh.scene().geometry)


def test_save_wavs(oyens_space: Space):
    # Add some microphones and sources
    oyens_space.add_microphone(microphone_type="ambeovr", keep_existing=False)    # just adds an ambeovr mic in a random plcae
    oyens_space.add_source()
    # Run the simulation
    oyens_space.simulate()
    # Dump the IRs to a temporary directory
    with TemporaryDirectory() as tmp:
        oyens_space.save_irs_to_wav(tmp)
        # We have 1 microphone with 4 capsules and 1 sound source
        #  We should have saved a WAV file for each of these
        for caps_idx in range(4):
            # The WAV file should exist
            fp = os.path.join(tmp, f"mic000_capsule00{caps_idx}_source000.wav")
            assert os.path.exists(fp)
            # Load up the WAV file in librosa and get the number of samples
            y, _ = sf.read(fp, )
            # Compare to the original IR
            x = oyens_space.irs["mic000"][caps_idx][0]
            assert np.allclose(y, x, atol=1e-4)
    # Temporary directory is implicitly cleaned up


@pytest.mark.parametrize(
    "microphone,sources,actual_doa",
    [
        # Test case 1: two sources at 90 and 270 degree angles from the mic
        (
            [-1.5, -1.5, 0.7],    # mic placed in bedroom 1
            [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
            [90, 270]
        ),
        # Test case 2: two sources at 0 and 180 degree angles from the mic
        (
            [2.9, -7.0, 0.3],    # mic placed in bedroom 2
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [0, 180]
        ),
        # Test case 3: combines 1 and 2 (four sources at 0, 90, 180, 270 degrees)
        (
            [2.5, 0., 0.5],     # mic placed in living room
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [-1.0, 0.0, 0.0], [0.0, -0.5, 0.0]],
            [0, 90, 180, 270]
        ),
        # Test case 4: single sound source at a 45-degree angle
        (
            [2.5, -1.0, 0.5],  # mic placed in living room
            [[1.0, 1.0, 0.0]],
            [45,]
        )
    ]
)
def test_simulated_doa_with_music(microphone: list, sources: list, actual_doa: list[int], oyens_space: Space):
    """
    Tests DOA of simulated sound sources and microphones with MUSIC algorithm.

    Places an Eigenmike32, simulates sound sources, runs MUSIC, checks that estimated DOA is near to actual DOA
    """
    # Add the microphones and simulate the space
    oyens_space.add_microphone(microphone_type="eigenmike32", position=microphone, keep_existing=False, alias="tester")
    oyens_space.add_sources(positions=sources, mic_aliases="tester", keep_existing=False)
    oyens_space.simulate()
    # TODO: in the future we should use simulated sound sources, not the IRs
    output = oyens_space.irs

    # Create the MUSIC object
    L = oyens_space.microphones["tester"].coordinates_absolute.T    # coordinates of our capsules for the eigenmike
    fs = int(oyens_space.ctx.config.sample_rate)
    nfft = 1024
    num_sources = len(oyens_space.sources)    # number of sound sources we've added
    assert num_sources == len(actual_doa) == len(sources)    # sanity check everything
    music = MUSIC(
        L=L,
        fs=fs,
        nfft=nfft,
        azimuth=np.deg2rad(np.arange(360)),
        num_sources=num_sources
    )

    # Iterating over all of our sound sources
    for doa_deg_true, source_idx in zip(actual_doa, range(num_sources)):
        # Get the IRs for this source: shape (N_capsules=32, 1=mono, N_samples)
        signals = np.vstack([m[:, source_idx, :] for m in output.values()])
        # Iterate over each individual IR (one per capsule: shape = 1, N_samples) and compute the STFT
        #  Stacked shape is (N_capsules, (N_fft / 2) + 1, N_frames)
        stft_signals = np.stack([stft(cs, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2] for cs in signals])
        # Sanity check the returned shape
        x, y, _ = stft_signals.shape
        assert x == oyens_space.microphones["tester"].n_capsules
        assert y == (nfft / 2) + 1
        # Run the music algorithm and get the predicted DOA
        music.locate_sources(stft_signals)
        doa_deg_pred = np.rad2deg(music.azimuth_recon[0])
        # Check that the predicted DOA is within a window of tolerance
        diff = abs(doa_deg_pred - doa_deg_true) % 360
        diff = min(diff, 360 - diff)  # smallest distance between angles
        assert diff <= 30


@pytest.mark.parametrize(
    "closemic_position,farmic_position,source_position",
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
    ]
)
def test_simulated_sound_distance(closemic_position: list, farmic_position: list, source_position: list, oyens_space):
    """
    Tests distance of simulated sound sources and microphones.

    Places a source and two AmbeoVR microphones near and far, then checks that the sound hits the close mic before far
    """

    # Add the microphones and simulate the space
    oyens_space.add_microphones(
        microphone_types=["ambeovr", AmbeoVR],
        positions=[closemic_position, farmic_position],
        aliases=["closemic", "farmic"],
        keep_existing=False
    )
    oyens_space.add_source(source_position, mic_alias="closemic", keep_existing=False)
    oyens_space.simulate()
    irs = oyens_space.irs
    # Shape of the IRs should be as expected
    assert len(irs) == 2
    # Get the IDX of the sample at which the sound hits both microphones
    arrival_close = min(np.flatnonzero(irs["mic000"]))
    arrival_far = min(np.flatnonzero(irs["mic001"]))
    # Should hit the closer mic before the further mic
    assert arrival_close < arrival_far
