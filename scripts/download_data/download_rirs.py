#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and prepares room impulse responses as .SOFA files. Modified from `SpatialScaper`."""

import argparse
import glob
import os
import random
import time
from pathlib import Path

import librosa
import netCDF4
import numpy as np
import pysofaconventions as pysofa
import soundfile as sf
from scipy.io import loadmat

from audiblelight import utils
from scripts.download_data.utils import combine_multizip, download_file, extract_zip

try:
    import mat73
except ImportError:
    raise ImportError(
        "'mat73' is required to download RIRs; install it with 'pip install mat73'"
    )


SAMPLE_RATE = 24000

DEFAULT_PATH = str(utils.get_project_root() / "resources/sofa")
DEFAULT_CLEANUP = True

METU_REMOTES = {
    "database_name": "metu",
    "remotes": {"spargair.zip": "https://zenodo.org/record/2635758/files/spargair.zip"},
}

TAU_REMOTES = {
    "database_name": "tau",
    "remotes": {
        "TAU-SRIR_DB.z01": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z01",
        "TAU-SRIR_DB.z02": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z02",
        "TAU-SRIR_DB.z03": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z03",
        "TAU-SRIR_DB.zip": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.zip",
        "TAU-SNoise_DB.z01": "https://zenodo.org/records/6408611/files/TAU-SNoise_DB.z01",
        "TAU-SNoise_DB.zip": "https://zenodo.org/records/6408611/files/TAU-SNoise_DB.zip",
    },
}

ARNI_REMOTES = {
    "database_name": "arni",
    "remotes": {
        "6dof_SRIRs_eigenmike_raw.zip": "https://zenodo.org/records/5720724/files/6dof_SRIRs_eigenmike_raw.zip",
        "6dof_SRIRs_eigenmike_SH.zip": "https://zenodo.org/records/5720724/files/6dof_SRIRs_eigenmike_SH.zip",
    },
}

MOTUS_REMOTES = {
    "database_name": "motus",
    "remotes": {
        "raw_rirs.zip": "https://zenodo.org/records/4923187/files/raw_rirs.zip",
        "sh_rirs.zip": "https://zenodo.org/records/4923187/files/sh_rirs.zip",
    },
}

RSOANU_REMOTES = {
    "database_name": "rsoanu",
    "remotes": {
        "RSoANU_RIRs_em32Eigenmike.zip": "https://zenodo.org/records/10720345/files/RSoANU_RIRs_em32Eigenmike.zip",
    },
}

DAGA_REMOTES = {
    "database_name": "daga",
    "remotes": {
        "DRIRs_Eigenmike_SOFAfiles.zip": "https://zenodo.org/records/2593714/files/DRIRs_Eigenmike_SOFAfiles.zip"
    },
}
GDRIVE_REMOTES = {
    "database_name": "gdrive",
    "remotes": {
        "daga_foa.sofa": "https://drive.google.com/uc?id=1Wa4XD9I_Xa7F2v_DitQtU4Zcru93Gnlf",
        "metu_foa.sofa": "https://drive.google.com/uc?id=1zamCd6OR6Tr5M40RdDhswYbT1wbGo2ZO",
        "rsoanu_foa.sofa": "https://drive.google.com/uc?id=1_EzzntIc_ypJ8MoLKreWGEhouWUKOCDY",
    },
}

NTAU_ROOMS = 9

METU_DB_NAME = "METU-SPARG"
TAU_DB_NAME = "TAU"
ARNI_DB_NAME = "ARNI"
MOTUS_DB_NAME = "MOTUS"
RSOANU_DB_NAME = "RSOANU"
DAGA_DB_NAME = "DAGA"

__TETRA_CHANS_IN_EM32__ = [5, 9, 25, 21]
__FOA_ACN_CHANS__ = [0, 1, 2, 3]


def load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt="mic", traj=None, flip=True):
    rooms = [
        "bomb_shelter",
        "gym",
        "pb132",
        "pc226",
        "sa203",
        "sc203",
        "se203",
        "tb103",
        "tc352",
    ]
    room = rooms[room_idx]
    files = os.listdir(tau_db_dir)
    rir_file = [file for file in files if room in file][0]
    rirs = mat73.loadmat(os.path.join(tau_db_dir, rir_file))["rirs"]
    output_paths, path_metadata, room_metadata = load_paths(room_idx, tau_db_dir)
    n_traj, n_heights = output_paths.shape
    n, r, _ = rirs[aud_fmt][0][0].shape
    path_stack = np.empty((0, 3))
    rir_stack = np.empty((n, r, 0))
    m = 0
    if traj is None:
        traj_iter = np.arange(n_traj)
    else:
        traj_iter = [traj]
    for i in traj_iter:
        for j in range(n_heights):
            path = output_paths[i, j]
            path_rirs = rirs[aud_fmt][i][j]
            if flip:
                if j % 2 == 1:
                    # flip every other height, as in DCASE
                    path_rirs = path_rirs[:, :, ::-1]
                    path = path[::-1]
            path_stack = np.concatenate((path_stack, path), axis=0)
            rir_stack = np.concatenate((rir_stack, path_rirs), axis=2)
            m += output_paths[i, j].shape[0]

    rirs = np.moveaxis(rir_stack, [0, 2], [2, 0])
    source_pos = path_stack

    assert rirs.shape == (m, r, n)
    assert source_pos.shape == (m, 3)
    mic_pos = np.repeat([room_metadata["microphone_position"]], m, axis=0)

    return rirs, source_pos, mic_pos, room


def create_srir_sofa(
    filepath,
    rirs,
    source_pos,
    mic_pos,
    db_name="Default_db",
    room_name="Room_name",
    listener_name="foa",
    sr=24000,
    comment="N/A",
):
    """
    Creates a SOFA file with spatial room impulse response data.

    This function generates a SOFA (Spatially Oriented Format for Acoustics) file to store spatial room impulse responses (SRIRs).
    It includes metadata about the recording environment, such as source and microphone positions, room characteristics, and listener details.

    Parameters:
        filepath (str): The path where the SOFA file will be created or overwritten.
        rirs (numpy.array): A 3D array of room impulse responses (measurements x receivers x samples).
        source_pos (numpy.array): The positions of the sound sources (measurements x coordinates).
        mic_pos (numpy.array): The positions of the microphones/listeners (measurements x coordinates).
        db_name (str, optional): Name of the database. Default is "Default_db".
        room_name (str, optional): Name of the room. Default is "Room_name".
        listener_name (str, optional): Name of the listener. Default is "foa".
        sr (int, optional): Sampling rate of the impulse responses. Default is 24000 Hz.
        comment (str, optional): Additional comments. Default is "N/A".

    Returns:
        None: The function does not return a value. It creates or overwrites a SOFA file at the specified filepath.
    """
    m_ = rirs.shape[0]
    r_ = rirs.shape[1]
    n_ = rirs.shape[2]
    e_ = 1
    i_ = 1
    c_ = 3

    assert rirs.shape == (m_, r_, n_)
    assert source_pos.shape == (m_, c_)

    # Need to delete it first if file already exists
    if os.path.exists(filepath):
        print(f"Overwriting {filepath}")
        os.remove(filepath)
    rootgrp = netCDF4.Dataset(filepath, "w", format="NETCDF4")

    # ----------Required Attributes----------#

    rootgrp.Conventions = "SOFA"
    rootgrp.Version = "2.1"
    rootgrp.SOFAConventions = "SingleRoomSRIR"
    rootgrp.SOFAConventionsVersion = "1.0"
    rootgrp.APIName = "pysofaconventions"
    rootgrp.APIVersion = "0.1.5"
    rootgrp.AuthorContact = "chris.ick@nyu.edu"
    rootgrp.Organization = "Music and Audio Research Lab - NYU"
    rootgrp.License = "Use whatever you want"
    rootgrp.DataType = "FIR"
    rootgrp.DateCreated = time.ctime(time.time())
    rootgrp.DateModified = time.ctime(time.time())
    rootgrp.Title = db_name + " - " + room_name
    rootgrp.RoomType = "shoebox"
    rootgrp.DatabaseName = db_name
    rootgrp.ListenerShortName = listener_name
    rootgrp.RoomShortName = room_name
    rootgrp.Comment = comment

    # ----------Required Dimensions----------#

    rootgrp.createDimension("M", m_)
    rootgrp.createDimension("N", n_)
    rootgrp.createDimension("E", e_)
    rootgrp.createDimension("R", r_)
    rootgrp.createDimension("I", i_)
    rootgrp.createDimension("C", c_)

    # ----------Required Variables----------#
    listener_position_var = rootgrp.createVariable("ListenerPosition", "f8", ("M", "C"))
    listener_position_var.Units = "metre"
    listener_position_var.Type = "cartesian"
    listener_position_var[:] = mic_pos

    listener_up_var = rootgrp.createVariable("ListenerUp", "f8", ("I", "C"))
    listener_up_var.Units = "metre"
    listener_up_var.Type = "cartesian"
    listener_up_var[:] = np.asarray([0, 0, 1])

    # Listener looking forward (+x direction)
    listener_view_var = rootgrp.createVariable("ListenerView", "f8", ("I", "C"))
    listener_view_var.Units = "metre"
    listener_view_var.Type = "cartesian"
    listener_view_var[:] = np.asarray([1, 0, 0])

    # single emitter for each measurement
    emitter_position_var = rootgrp.createVariable(
        "EmitterPosition", "f8", ("E", "C", "I")
    )
    emitter_position_var.Units = "metre"
    emitter_position_var.Type = "spherical"
    # Equidistributed speakers in circle
    emitter_position_var[:] = np.zeros((e_, c_, i_))

    source_position_var = rootgrp.createVariable("SourcePosition", "f8", ("M", "C"))
    source_position_var.Units = "metre"
    source_position_var.Type = "cartesian"
    source_position_var[:] = source_pos

    source_up_var = rootgrp.createVariable("SourceUp", "f8", ("I", "C"))
    source_up_var.Units = "metre"
    source_up_var.Type = "cartesian"
    source_up_var[:] = np.asarray([0, 0, 1])

    source_view_var = rootgrp.createVariable("SourceView", "f8", ("I", "C"))
    source_view_var.Units = "metre"
    source_view_var.Type = "cartesian"
    source_view_var[:] = np.asarray([1, 0, 0])

    receiver_position_var = rootgrp.createVariable(
        "ReceiverPosition", "f8", ("R", "C", "I")
    )
    receiver_position_var.Units = "metre"
    receiver_position_var.Type = "cartesian"
    receiver_position_var[:] = np.zeros((r_, c_, i_))

    sampling_rate_var = rootgrp.createVariable("Data.SamplingRate", "f8", ("I",))
    sampling_rate_var.Units = "hertz"
    sampling_rate_var[:] = sr

    delay_var = rootgrp.createVariable("Data.Delay", "f8", ("I", "R"))
    delay = np.zeros((i_, r_))
    delay_var[:, :] = delay

    data_ir_var = rootgrp.createVariable("Data.IR", "f8", ("M", "R", "N"))
    data_ir_var.ChannelOrdering = "acn"  # standard ambi ordering
    data_ir_var.Normalization = "sn3d"
    data_ir_var[:] = rirs

    # ----------Close it----------#

    rootgrp.close()
    print(f"SOFA file saved to {filepath}")


def unitvec_to_cartesian(path_unitvec, height, dist):
    if isinstance(dist, np.ndarray):
        z_offset = height
        rad = np.sqrt(dist[0] ** 2 + (dist[2] + z_offset) ** 2)
        scaled_path = map_to_cylinder(path_unitvec, rad, axis=1)
    else:
        scaled_path = map_to_cylinder(path_unitvec, dist, axis=2)
    return scaled_path


def map_to_cylinder(path, rad, axis=2):
    # maps points (unit vecs) to cylinder of known radius along axis (default z/2)
    scaled_path = np.empty(path.shape)
    rad_axes = [0, 1, 2]
    rad_axes.remove(axis)
    for i in range(path.shape[0]):
        vec = path[i]
        scale_rad = np.sqrt(np.sum([vec[j] ** 2 for j in rad_axes]))
        scale = rad / scale_rad
        scaled_path[i] = vec * scale
    return scaled_path


def load_paths(room_idx, tau_db_dir, center_on_mic=False):
    rooms = [
        "bomb_shelter",
        "gym",
        "pb132",
        "pc226",
        "sa203",
        "sc203",
        "se203",
        "tb103",
        "tc352",
    ]
    room = rooms[room_idx]

    measinfo = loadmat(os.path.join(tau_db_dir, "measinfo.mat"))["measinfo"]
    rirdata = loadmat(os.path.join(tau_db_dir, "rirdata.mat"))["rirdata"][0]

    trajs = measinfo[room_idx][0][4][0]
    heights = measinfo[room_idx][0][5][0]
    dists = measinfo[room_idx][0][6][0]
    mic_pos = measinfo[room_idx][0][7][0]
    traj_type = measinfo[room_idx][0][9][0]
    paths = rirdata[0][1][room_idx][0][2]

    output_paths = np.empty(paths.shape, dtype=object)
    path_metadata = np.empty(paths.shape, dtype=object)
    room_metadata = {
        "room": room,
        "trajectory_type": traj_type,
        "microphone_position": mic_pos,
    }
    for i, traj in enumerate(trajs):
        for j, height in enumerate(heights):
            if traj_type == "circular":
                dist = dists[i]
            elif traj_type == "linear":
                dist = dists[:, i]
            else:
                raise ValueError(f"Unknown trajectory type: {traj_type}")

            path_unitvec = paths[i, j][0]
            path_dict = {"trajectory": traj, "height": height}
            path_cartesian = unitvec_to_cartesian(path_unitvec, height, dist)
            if center_on_mic:
                path_cartesian += mic_pos
            output_paths[i, j] = path_cartesian
            path_metadata[i, j] = path_dict

    return output_paths, path_metadata, room_metadata


def create_single_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name):
    db_dir = sofa_db_dir
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    for room_idx in range(NTAU_ROOMS):
        # Load flattened (and flipped) rirs/paths from TAU-SRIR database
        rirs, source_pos, mic_pos, room = load_flat_tau_srir(
            tau_db_dir, room_idx, aud_fmt=aud_fmt
        )

        filepath = os.path.join(db_dir, f"{room}_{aud_fmt}.sofa")
        comment = f"SOFA conversion of {room} from TAU-SRIR-DB"

        print(
            f"Creating .sofa file for {aud_fmt}, Room: {room} (Progress: {room_idx + 1}/9)"
        )

        # Create .sofa files with flattened rirs/paths + metadata
        create_srir_sofa(
            filepath,
            rirs,
            source_pos,
            mic_pos,
            db_name=db_name,
            room_name=room,
            listener_name=aud_fmt,
            sr=SAMPLE_RATE,
            comment=comment,
        )


def download_and_extract_remotes(urls_dict, extract_to, cleanup=False):

    extract_to.mkdir(parents=True, exist_ok=True)
    if not os.listdir(extract_to):
        # Extract the filename from the URL
        for filename, url in urls_dict.items():
            # Check if the zip file already exists
            zip_path = extract_to / filename
            zip_extract_to = zip_path.parent
            zip_extract_to.mkdir(parents=True, exist_ok=True)
            if zip_path.is_file():
                print(f"Zip file {zip_path} already present. Skipping download.")
            else:
                download_file(url, zip_path)
                # Download and extract the file
                extract_zip(zip_path, zip_extract_to)
                # remove the zip file after extraction
                if cleanup:
                    os.remove(zip_path)
    else:
        print(
            f"Data already present in directory {extract_to}. Skipping database download."
        )


def prepare_metu(dataset_path, dest_path_sofa):
    spargpath = Path(dataset_path) / "spargair" / "em32"
    xy_zs = os.listdir(spargpath)

    def convert(xyz__):
        x_, y_, z_ = xyz__
        _x__ = (3 - int(x_)) * 0.5
        _y__ = (3 - int(y_)) * 0.5
        _z__ = (int(z_) - 2) * 0.5
        return _x__, _y__, _z__

    i_rs = []
    xyzs = []
    sr = None

    for XYZ in xy_zs:
        xyz = convert(XYZ)
        xyzs.append(xyz)

        wavpath = spargpath / XYZ
        x__ = []
        for ichan in __TETRA_CHANS_IN_EM32__:  # Specific channels for 'mic' format
            wavfile = wavpath / f"IR{ichan+1:05d}.wav"
            x, sr = sf.read(wavfile)
            x__.append(x)
        i_rs.append(np.array(x__))

    filepath = dest_path_sofa / "metu_mic.sofa"
    rirs = np.array(i_rs)
    source_pos = np.array(xyzs)
    mic_pos = np.array([[0, 0, 0]])

    create_srir_sofa(
        filepath,
        rirs,
        source_pos,
        mic_pos,
        db_name=METU_DB_NAME,
        room_name="classroom",
        listener_name="em32",
        sr=sr,
    )


def prepare_motus(dataset_path, dest_path_sofa, audio_fmts: list[str] = None):
    if audio_fmts is None:
        audio_fmts = ["foa", "mic"]
    source_positions = {
        "1": np.array([[1.637, 0.0, 0.0]]),
        "2": np.array([[-0.078, 1.663, 0.0]]),
        "3": np.array([[0.658, 1.22, 0.0]]),
        "4": np.array([[2.056, 1.362, 0.0]]),
    }
    mic_pos = np.array([[0.0, 0.0, 0.0]])
    sr = None

    for fmt in audio_fmts:
        rir_file_names = os.listdir(dataset_path)
        if fmt == "foa":
            rir_file_names = [f for f in rir_file_names if "sh" in f]
        elif fmt == "mic":
            rir_file_names = [f for f in rir_file_names if "raw" in f]
        irs, xyzs = [], []
        for filename in rir_file_names:
            source_pos_index = filename.split("_")[1]
            source_pos = source_positions[source_pos_index] + random.uniform(
                -0.001, 0.001
            )  # mm
            xyzs.append(source_pos)
            wavfile = dataset_path / filename
            x, sr = sf.read(wavfile)
            if fmt == "foa":
                x = (x[:, __FOA_ACN_CHANS__]).T
            elif fmt == "mic":
                x = (x[:, __TETRA_CHANS_IN_EM32__]).T
            irs.append(x)
        rirs = np.array(irs)
        # Reshape source_pos to have shape (num_measurements, 3) as required
        # (M,C) aka... (num measurements, num coordinates)
        source_pos = np.array(xyzs).reshape(len(xyzs), 3)
        filepath = dest_path_sofa / f"motus_{fmt}.sofa"
        create_srir_sofa(
            filepath,
            rirs,
            source_pos,
            mic_pos,
            db_name=MOTUS_DB_NAME,
            room_name="motus",
            listener_name=fmt,
            sr=sr,
        )


def prepare_rsoanu(dataset_path, dest_path_sofa, audio_fmts: list[str] = None):
    if audio_fmts is None:
        audio_fmts = ["mic"]

    source_positions = {
        "1": np.array([6.75, 3.75, 1.2]),
        "2": np.array([4.75, 4.25, 1.384]),
        "3": np.array([2.25, 2.50, 0.93]),
    }
    sr = None
    for fmt in audio_fmts:
        datapath = Path(dataset_path) / "RSoANU_RIRs_em32Eigenmike"
        irs, xyzs = [], []
        for folder in os.scandir(datapath):
            if not folder.is_dir():
                continue
            wav_files_path = Path(folder) / "WAV Files"
            sr = process_wav_files(wav_files_path, fmt, source_positions, irs, xyzs)
        rirs = np.array(irs)
        source_pos = np.array(xyzs).reshape(len(xyzs), 3)
        filepath = dest_path_sofa / f"rsoanu_{fmt}.sofa"
        mic_pos = np.zeros((len(source_pos), 3))
        create_srir_sofa(
            filepath,
            rirs,
            source_pos,
            mic_pos,
            db_name=RSOANU_DB_NAME,
            room_name="rsoanu",
            listener_name=fmt,
            sr=sr,
        )


def process_wav_files(wav_files_path, fmt, source_positions, irs, xyzs):
    sr = None
    for filename in os.scandir(wav_files_path):
        if not filename.name.endswith(".wav"):
            continue
        source_pos_index = filename.name[5]
        x, y = parse_coordinates(filename.name)
        mic_pos = np.array([x, y, 1.7])
        source_pos = (source_positions[source_pos_index] - mic_pos) + random.uniform(
            -0.001, 0.001
        )
        xyzs.append(source_pos)
        x, sr = sf.read(filename.path)
        if fmt == "foa":
            pass
        elif fmt == "mic":
            x = (x[:, __TETRA_CHANS_IN_EM32__]).T
        irs.append(x)
    return sr


def parse_coordinates(filename):
    if filename[8] == "e":
        x = (int(filename[12:14]) * 0.1) + 1.25

        y = 8.5 - ((int(filename[9:11]) * 0.1) + 0.75)
    else:
        x = int(filename[12]) + 1.25
        y = 8.5 - (int(filename[9]) + 0.75)
    return x, y


def download_tau(dest_path, urls, cleanup=False):
    # Download combine and extract zip files
    dest_path.mkdir(parents=True, exist_ok=True)
    for filename, url in urls.items():
        download_file(url, dest_path / filename)
    combine_multizip(f"{dest_path/'TAU-SRIR_DB.zip'}", f"{dest_path/'single.zip'}")
    extract_zip(dest_path / "single.zip", dest_path)
    combine_multizip(f"{dest_path/'TAU-SNoise_DB.zip'}", f"{dest_path/'single.zip'}")
    extract_zip(dest_path / "single.zip", dest_path)
    if cleanup:
        for root, dirs, files in os.walk(dest_path):
            # Find all zip files in current directory
            for file in glob.glob(os.path.join(root, "*.z*")):
                # Remove the zip file
                os.remove(file)


def prepare_tau(path_raw, path_sofa, audio_formats: list[str] = None):
    if audio_formats is None:
        audio_formats = ["foa", "mic"]

    # generate Sofa files
    tau_db_dir = f"{path_raw/'TAU-SRIR_DB'}"
    sofa_db_dir = path_sofa
    for aud_fmt in audio_formats:
        print(f"Starting .sofa creation for {aud_fmt} format.")
        create_single_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, TAU_DB_NAME)
        print(f"Finished .sofa creation for {aud_fmt} format.")


def get_absorption_level_arni(filename):
    return int(filename.split("_")[4].replace("percent", ""))


def center_and_translate_arni(receiver_pos, source_pos):
    # Given two points, center the receiver coordinate at zero and tranlate the source
    y1, x1, z1 = receiver_pos[0], receiver_pos[1], receiver_pos[2]
    y2, x2, z2 = source_pos[0], source_pos[1], source_pos[2]
    # compute translation of the source (loudspeaker)
    # add small perturbation to have unique coordinate for trajectory generation purposes
    translation_y = -y1 + random.uniform(-0.001, 0.001)
    translation_x = -x1 + random.uniform(-0.001, 0.001)
    translation_z = z1 + random.uniform(-0.001, 0.001)
    # apply tranlation, note that the receiver (mic) remains at the same height
    receiver_centered = [0, 0, 0]
    source_translated = [x2 + translation_x, y2 + translation_y, translation_z - z2]
    return receiver_centered, source_translated


def create_single_sofa_file_arni(aud_fmt, arni_db_dir, sofa_db_dir, room="ARNI"):
    # Ensure the output directory exists
    if not os.path.exists(sofa_db_dir):
        os.makedirs(sofa_db_dir)

    # Gather all .sofa files in the source directory
    sofa_files = [file for file in os.listdir(arni_db_dir) if file.endswith(".sofa")]
    if not sofa_files:
        raise ValueError(f"Error: {arni_db_dir} contains no .sofa files")

    # Create the output file path
    filepath = os.path.join(sofa_db_dir, f"arni_{aud_fmt}.sofa")

    # Initialize containers for source and microphone positions, and RIRs
    source_positions, mic_positions, rirs = [], [], []

    # Process each sofa file
    for sofa_file in sorted(
        sofa_files, key=lambda x: get_absorption_level_arni(x)
    ):  # Sort by absorption level
        sofa_path_ = os.path.join(arni_db_dir, sofa_file)
        sofa = pysofa.SOFAFile(sofa_path_, "r")

        if not sofa.isValid():
            print("Error: the file is invalid")
            break

        # Extract data from the SOFA object
        source_position = sofa.getVariableValue("SourcePosition")
        listener_position = sofa.getVariableValue("ListenerPosition")
        rir_data = sofa.getDataIR()

        # Assuming a fixed number of measurements to simplify
        num_measurements = 21  # Number of measurements to consider

        # Loop through each measurement
        for i in range(num_measurements):
            ir_data = rir_data[i, :]
            ir_data_resampled = librosa.resample(
                ir_data, orig_sr=48000, target_sr=SAMPLE_RATE
            )

            # Append data depending on the audio format
            if aud_fmt == "mic":
                rirs.append(
                    ir_data_resampled[__TETRA_CHANS_IN_EM32__, :]
                )  # Specific channels for 'mic' format
            else:  # 'foa' format
                rirs.append(ir_data_resampled[:4])

            # Compute the centered and translated positions
            mic_centered, src_translated = center_and_translate_arni(
                listener_position[i], source_position[i]
            )
            mic_positions.append(mic_centered)
            source_positions.append(src_translated)

    # Convert lists to numpy arrays
    rirs = np.array(rirs)
    mic_positions = np.array(mic_positions)
    source_positions = np.array(source_positions)

    # Create .sofa file
    create_srir_sofa(
        filepath,
        rirs,
        source_positions,
        mic_positions,
        room_name=room,
        listener_name=aud_fmt,
        sr=SAMPLE_RATE,
        comment=f"SOFA conversion of room {room}",
    )


def prepare_arni(path_raw, path_sofa, audio_formats: list[str] = None):
    if audio_formats is None:
        audio_formats = ["mic", "foa"]

    # generate sofa files
    sofa_db_dir = path_sofa
    for aud_fmt in audio_formats:
        if aud_fmt == "mic":
            arni_db_dir = f"{path_raw}/6dof_SRIRs_eigenmike_raw"
        elif aud_fmt == "foa":
            arni_db_dir = f"{path_raw}/6dof_SRIRs_eigenmike_SH"
        else:
            raise ValueError(f"Unknown audio format: {aud_fmt}")
        print(f"Starting .sofa creation for {aud_fmt} format.")
        create_single_sofa_file_arni(aud_fmt, arni_db_dir, sofa_db_dir, ARNI_DB_NAME)
        print(f"Finished .sofa creation for {aud_fmt} format.")


def prepare_daga(source_path_, sofa_path_, audio_formats: list[str] = None):
    if audio_formats is None:
        audio_formats = ["mic"]

    # the source positions are exactly in front of the microphone
    source_positions = {"0": np.array([2.5, 0, 0]), "180": np.array([2.8, 0, 0])}

    sofa_folder = Path(source_path_)
    dest_path_sofa = Path(sofa_path_)
    aggregated_irs = {fmt: [] for fmt in audio_formats}
    aggregated_source_positions = {fmt: [] for fmt in audio_formats}

    orig_sr = None

    for sofa_file in sofa_folder.glob("*.sofa"):

        source_angle = "180" if "180" in sofa_file.name else "0"
        source_pos = source_positions[source_angle] + random.uniform(-0.001, 0.001)

        sofa = netCDF4.Dataset(sofa_file, "r")
        irs = sofa.variables["Data.IR"][:]
        orig_sr = sofa.variables["Data.SamplingRate"][:][0]
        sofa.close()

        for fmt in audio_formats:
            if fmt == "mic":
                processed_irs = irs[:, __TETRA_CHANS_IN_EM32__, 0].T
            else:
                print(f"Format {fmt} not supported, skipping!")
                continue

            aggregated_irs[fmt].append(processed_irs)
            aggregated_source_positions[fmt].append(source_pos)

    for fmt in audio_formats:
        if fmt != "mic":
            continue

        all_irs = np.stack(aggregated_irs[fmt], axis=0)
        all_source_positions = np.array(aggregated_source_positions[fmt])

        filepath = dest_path_sofa / f"daga_{fmt}.sofa"

        create_srir_sofa(
            filepath,
            all_irs,
            all_source_positions,
            np.zeros_like(all_source_positions),
            db_name=DAGA_DB_NAME,
            room_name="daga",
            listener_name=fmt,
            sr=orig_sr,
        )


def download_gdrive(source_path, remote_path):
    import gdown

    gdown.download(remote_path, str(source_path))


def main(path: Path = DEFAULT_PATH, cleanup: bool = DEFAULT_CLEANUP):
    f"""
    Downloads and prepares room impulse responses as .SOFA files. Modified from `SpatialScaper`.

    Arguments:
        path: Path to store and process the dataset, defaults to {DEFAULT_PATH}.
        cleanup: Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}.
    """
    print("---- RIR download script ----")
    print(f"RIRs will be downloaded to: {path}")

    source_path, sofa_path = (
        Path(args.path) / "source_data",
        Path(args.path) / "rirs",
    )
    for p in (source_path, sofa_path):
        os.makedirs(p, exist_ok=True)

    # Google Drive (hosts a few FOA files not converted here)
    for fname, rem in GDRIVE_REMOTES["remotes"].items():
        outpath = sofa_path / fname
        download_gdrive(outpath, rem)

    # METU
    metu_path = source_path / METU_REMOTES["database_name"]
    download_and_extract_remotes(METU_REMOTES["remotes"], metu_path, cleanup)
    prepare_metu(metu_path, sofa_path)

    # TAU
    tau_path = source_path / TAU_REMOTES["database_name"]
    download_tau(tau_path, TAU_REMOTES["remotes"], cleanup)
    prepare_tau(tau_path, sofa_path)

    # ARNI
    arni_path = source_path / ARNI_REMOTES["database_name"]
    download_and_extract_remotes(ARNI_REMOTES["remotes"], arni_path, cleanup)
    prepare_arni(arni_path, sofa_path)

    # MOTUS
    motus_path = source_path / MOTUS_REMOTES["database_name"]
    download_and_extract_remotes(MOTUS_REMOTES["remotes"], motus_path, cleanup)
    prepare_motus(motus_path, sofa_path)

    # RSOANU
    rsoanu_path = source_path / RSOANU_REMOTES["database_name"]
    download_and_extract_remotes(RSOANU_REMOTES["remotes"], rsoanu_path, cleanup)
    prepare_rsoanu(rsoanu_path, sofa_path)

    # DAGA DRIR
    daga_path = source_path / DAGA_REMOTES["database_name"]
    download_and_extract_remotes(DAGA_REMOTES["remotes"], daga_path, cleanup)
    prepare_daga(daga_path, sofa_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to store and process the dataset. Defaults to {DEFAULT_PATH}",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=f"Whether to cleanup after download. Defaults to {DEFAULT_CLEANUP}",
    )
    args = parser.parse_args()

    main(args.path, args.cleanup)
