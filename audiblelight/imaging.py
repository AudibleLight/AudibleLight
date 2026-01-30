#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules and functions for generating acoustic images from multichannel audio recordings

By default, we use linear spacing for frequency bands ranging from 1500Hz to 4500Hz, with a total of 9 bands. We
use a timescale of 100 ms to match the labelling resolution of the DCASE files.

Much of the code is adapted from [this repo](https://github.com/adrianSRoman/LAM/tree/main/dataset/gen_dataset)
"""

import math
import time
from typing import Callable, Optional

import astropy.coordinates as coord
import astropy.units as u
import librosa
import numpy as np
import pyunlocbox as opt
import scipy.linalg as linalg
import scipy.signal.windows as windows
import scipy.sparse.linalg as splinalg
import skimage.util as skutil
from pyunlocbox.functions import dummy
from scipy.constants import speed_of_sound
from scipy.interpolate import griddata
from tqdm import tqdm

from audiblelight import config, custom_types, utils


class L2Loss(opt.functions.func):
    """
    L2 loss function
    """

    def __init__(self, s: np.ndarray, a: np.ndarray):
        m, n = a.shape
        if not ((s.shape[0] == s.shape[1]) and (s.shape[0] == m)):
            raise ValueError("Parameters `s` and `a` are inconsistent.")
        if not np.allclose(s, s.conj().T):
            raise ValueError("Parameter `s` must be Hermitian.")

        super().__init__()
        self._S = s.copy()
        self._A = a.copy()

    def _eval(self, x: np.ndarray) -> custom_types.Numeric:
        """
        Function evaluation.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        m, n = self._A.shape
        q = x.shape[1]
        b = (
            (self._A.reshape(1, m, n) * x.reshape(n, 1, q).T) @ self._A.conj().T
        ) - self._S

        z = np.sum(b * b.conj()).real
        return z

    def _grad(self, x: np.ndarray) -> np.ndarray:
        """
        Function gradient.
        """
        was_1d = x.ndim == 1
        if was_1d:
            x = x.reshape(-1, 1)

        m, n = self._A.shape
        q = x.shape[1]
        b = (
            (self._A.reshape(1, m, n) * x.reshape(n, 1, q).T) @ self._A.conj().T
        ) - self._S

        z = 2 * np.sum(self._A.conj() * (b @ self._A), axis=1).real.T
        if was_1d:
            z = z.reshape(-1)
        return z


class ElasticNetLoss(opt.functions.func):
    """
    Elastic-net regularizer.
    """

    def __init__(self, lambda_: custom_types.Numeric, gamma: custom_types.Numeric):
        if lambda_ < 0:
            raise ValueError("Parameter `lambda_` must be positive.")
        if not (0 <= gamma <= 1):
            raise ValueError("Parameter `gamma` must be in (0, 1).")

        super().__init__()
        self._lambda = lambda_
        self._gamma = gamma

    def _eval(self, x: np.ndarray) -> custom_types.Numeric:
        """
        Function evaluation.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        l1_term = self._gamma * np.sum(np.abs(x), axis=0)
        l2_term = (1 - self._gamma) * np.sum(x**2, axis=0)

        z = np.sum(self._lambda * (l1_term + l2_term))
        return z

    def _prox(self, x, alpha):
        """
        Function proximal operator.
        """
        c1 = self._lambda * alpha * self._gamma
        c2 = 2 * self._lambda * alpha * (1 - self._gamma) + 1

        z = np.clip((x - c1) / c2, a_min=0, a_max=None)
        return z


class GroundTruthAccel(opt.acceleration.accel):
    """
    Acceleration scheme used to evaluate Acoustic Camera ground-truth.
    """

    def __init__(
        self, d: custom_types.Numeric, l_: custom_types.Numeric, momentum: bool = True
    ):
        super().__init__()

        if d < 2:
            raise ValueError("Parameter `d` is out of range.")

        self._d = d
        self._step = 1 / l_
        self._sol_prev = 0
        self._momentum = momentum

    def _pre(self, functions, x0):
        """
        Pre-processing specific to the acceleration scheme.
        """
        pass

    def _update_step(self, solver, objective, niter):
        """
        Update the step size for the next iteration.
        """
        return self._step

    def _update_sol(self, solver, objective, niter):
        """
        Update the solution point for the next iteration.
        """
        if self._momentum:
            step = (niter - 1) / (niter + self._d)
            sol = solver.sol + step * (solver.sol - self._sol_prev)
        else:
            sol = solver.sol
        self._sol_prev = solver.sol
        return sol

    def _post(self):
        """
        Post-processing specific to the acceleration scheme.
        """
        pass


def _convert_mic_coordinates(mic_coords: np.ndarray) -> dict[str, list]:
    """
    Convert the microphone coordinate format normally used in AudibleLight to the format used in the LAM paper code.

    By default, AudibleLight uses polar coordinates given as azimuth, elevation, radius:
        - Azimuth is measured counter-clockwise in degrees between 0 and 360, where 0 == the front of the microphone.
        - Elevation is measured between -90 and 90 degrees, where 0 == "straight ahead", 90 == "up", -90 == "down".
        - Radius is measured in meters away from the center of the array.

    LAM instead uses colatitude, azimuth, radius:
        - Colatitude is calculated between 0 and 180 degrees, where 0 == straight up, 90 == straight ahead, 180 == down
        - Azimuth is measured between 0 and 360 degrees, where 0 == the front of the microphone
        - Radius is measured as before.

    Arguments:
        mic_coords (np.ndarray): this should be polar coordinates, e.g. MicArray.coordinates_polar

    Returns:
        dict: converted microphone coordinates
    """
    # Make a copy to start so we don't modify the initial array
    coord_lam = mic_coords.copy()

    # First, swap columns one and two
    coord_lam[:, 0], coord_lam[:, 1] = coord_lam[:, 1], coord_lam[:, 0].copy()

    # Next, we can simply minus 90 to get colatitude
    coord_lam[:, 0] = 90 - coord_lam[:, 0]

    # We need to do something more complex to get azimuth
    coord_lam[:, 1] = np.where(
        coord_lam[:, 1] < 0, coord_lam[:, 1] + 360, coord_lam[:, 1]
    )

    # Radius and shape remains unchanged
    # assert np.array_equal(coord_lam[:, 2], mic_coords[:, 2])
    # assert coord_lam.shape == mic_coords.shape

    # LAM paper uses a dictionary coordinate system
    return {
        str(n): [int(v[0]), int(v[1]), float(v[2])] for n, v in enumerate(coord_lam, 1)
    }


def _degrees_to_radians(coords_dict: dict[str, list]) -> dict[str, list]:
    """
    Take a dictionary with microphone array capsules and 3D polar coordinates to convert them from degrees to radians
    colatitude, azimuth, and radius (radius is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }


def _polar_to_cartesian(coords_dict: dict[str, list], units: Optional[str] = None):
    """
    Take a dictionary with microphone array capsules and polar coordinates and convert to cartesian
    """
    if (
        units is None
        or not isinstance(units, str)
        or units.lower() not in ["degrees", "radians"]
    ):
        raise ValueError("Units must be specified as one of 'degrees' or 'radians'")
    elif units.lower() == "degrees":
        coords_dict = _degrees_to_radians(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }


def _equirectangular_to_cartesian(
    r: custom_types.Numeric, lat: custom_types.Numeric, lon: custom_types.Numeric
) -> np.ndarray:
    """
    Convert equirectangular values in form radius, latitude, longitude to cartesian
    """
    r = np.array([r])

    # Must be non-negative
    if np.any(r < 0):
        raise ValueError("Parameter `r` must be non-negative.")

    return (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )


def _cartesian_to_spherical(
    x: custom_types.Numeric, y: custom_types.Numeric, z: custom_types.Numeric
) -> tuple[custom_types.Numeric, custom_types.Numeric]:
    """
    Convert Cartesian (x, y, z) to spherical (azimuth, elevation) in degrees.
    """
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z))
    return azimuth, elevation


def _spherical_to_equirectangular(
    azimuth_deg: custom_types.Numeric,
    elevation_deg: custom_types.Numeric,
    width: custom_types.Numeric,
    height: custom_types.Numeric,
) -> tuple[custom_types.Numeric, custom_types.Numeric]:
    """
    Convert spherical coordinates to equirectangular pixel coordinates.

    Arguments:
        azimuth_deg: Azimuth in degrees [-180, 180]
        elevation_deg: Elevation in degrees [-90, 90]
        width: Width in pixels
        height: Height in pixels

    Returns:
        (x, y) pixel coordinates
    """
    # normalise azimuth from [-180, 180] to [0, img_width]
    #  azimuth 0° should be at centre (x = img_width/2)
    #  azimuth -180° should be at left edge (x = 0)
    #  azimuth +180° should be at right edge (x = img_width)
    x = ((-azimuth_deg + 180) % 360) / 360.0 * width

    # normalise elevation from [-90, 90] to [0, img_height]
    #  elevation +90° (up) should be at top (y = 0)
    #  elevation -90° (down) should be at bottom (y = img_height)
    y = (90 - elevation_deg) / 180.0 * height

    return int(x), int(y)


def _equirectangular_to_spherical(
    x: custom_types.Numeric,
    y: custom_types.Numeric,
    width: custom_types.Numeric,
    height: custom_types.Numeric,
) -> tuple[custom_types.Numeric, custom_types.Numeric]:
    """
    Convert equirectangular pixel coordinates back to spherical coordinates.

    Arguments:
        x: Pixel x-coordinate
        y: Pixel y-coordinate
        width: Width in pixels
        height: Height in pixels

    Returns:
        (azimuth_deg, elevation_deg)
    """
    azimuth_deg = 180.0 - (x / width) * 360.0
    elevation_deg = 90.0 - (y / height) * 180.0
    return azimuth_deg, elevation_deg


def get_mic_xyz_coords(mic_coords: dict) -> list:
    """
    Get XYZ coordinates for microphone array
    """
    mic_coords_conv = _polar_to_cartesian(mic_coords, units="degrees")
    xyz = [[coo_ for coo_ in mic_coords_conv[ch]] for ch in mic_coords_conv]
    return xyz


def fibonacci(
    n: custom_types.Numeric,
    direction: Optional[np.ndarray] = None,
    fo_v: Optional[custom_types.Numeric] = None,
) -> np.ndarray:
    """
    Generate points on a unit sphere using Fibonacci lattice sampling.

    The Fibonacci lattice provides a nearly uniform distribution of points on a sphere's surface, making it ideal for
    spherical sampling applications. Points can optionally be limited to a specific region defined by a direction
    vector and field of view.

    Arguments:
        n (Numeric): Refinement level that determines the number of points. The total number of points generated is
            `4 * (n + 1)^2`.
        direction (np.ndarray, optional): A 3D unit vector specifying the central direction for region-limited
            sampling. Must be provided together with `fo_v`. If None, generates points over the entire sphere.
        fo_v (Numeric, optional): Field of view in radians, defining the angular extent of the region around
            `direction`. Must be in the range (0, 2π) or equivalently (0, 360) degrees. Required if `direction` is
            specified.

    Returns:
        np.ndarray: Array of shape (3, m) containing the Cartesian coordinates (x, y, z) of points on the unit sphere,
            where m ≤ 4 * (n + 1)^2. When region-limited, m is reduced to include only points within the specified FOV.
    """

    def _pol2cart(r, col, lo):
        lat = (np.pi / 2) - col
        return _equirectangular_to_cartesian(r, lat, lo)

    # This is the type of tesselation that we are using
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if fo_v is not None:
            if not (0 < np.rad2deg(fo_v) < 360):
                raise ValueError("Parameter `fo_v` must be in (0, 360) degrees.")
        else:
            raise ValueError(
                "Parameter `fo_v` must be specified if `direction` is provided."
            )

    if n < 0:
        raise ValueError("Parameter `n` must be non-negative.")

    n_px = 4 * (n + 1) ** 2
    n = np.arange(n_px)

    colat = np.arccos(1 - (2 * n + 1) / n_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    xyz = np.stack(_pol2cart(1, colat, lon), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(fo_v / 2)
        mask = (direction @ xyz) >= min_similarity
        xyz = xyz[:, mask]

    # these are the cartesian coordinates of the tesselation
    #  need to turn this into azimuth + elevation
    #  need to do the inverse of this: cart2pol
    #  sphere will have fewer points at the poles than expected
    #  to fill these, we need to do another interpolation
    return xyz


def get_field(
    sh_order: Optional[custom_types.Numeric] = config.AIMG_SH_ORDER,
) -> np.ndarray:
    """
    Generate a hemisphere of sampling points for spherical harmonic field visualization.

    Creates a Fibonacci lattice on a unit sphere and filters it to retain only points within the upper hemisphere
    (i.e., z ≥ 0), with additional border trimming to avoid edge artifacts in visualization or processing.

    Arguments:
        sh_order (Numeric): Spherical harmonic order that determines sampling density. Higher orders produce more
            points. Defaults to `config.AIMG_SH_ORDER`. The initial grid contains `4 * (sh_order + 1)^2` points before
            filtering.

    Returns:
        np.ndarray: Array of shape (3, n) containing Cartesian coordinates (x, y, z) of points on the upper hemisphere.
    """

    # generate lattice
    r = fibonacci(sh_order)
    r_mask = np.abs(r[2, :]) < np.sin(np.deg2rad(90))
    r = r[:, r_mask]  # Shrink visible view to avoid border effects.
    # this is cartesian coordinates: (3, n_px)
    return r


def steering_operator(
    xyz: np.ndarray,
    r: custom_types.Numeric,
    fmin: Optional[custom_types.Numeric] = config.AIMG_FMIN,
    fmax: Optional[custom_types.Numeric] = config.AIMG_FMAX,
    n_bands: Optional[custom_types.Numeric] = config.AIMG_NBANDS,
) -> np.ndarray:
    """
    Compute steering matrix.
    """
    freq = np.linspace(fmin, fmax, n_bands)
    wl = speed_of_sound / (freq.max() + 500)
    if wl <= 0:
        raise ValueError(f"Parameter `wl` must be positive (got {wl}).")

    scale = 2 * np.pi / wl
    return np.exp((-1j * scale * xyz.T) @ r)


def extract_visibilities(
    data_: np.ndarray,
    rate_: custom_types.Numeric,
    t: custom_types.Numeric,
    fc: custom_types.Numeric,
    bw: custom_types.Numeric,
    alpha: custom_types.Numeric,
) -> np.ndarray:
    """
    Transform time-series to visibility matrices.
    """
    n_stft_sample = int(rate_ * t)
    if n_stft_sample == 0:
        raise ValueError("Not enough samples per time frame.")

    n_sample = (data_.shape[0] // n_stft_sample) * n_stft_sample
    n_channel = data_.shape[1]
    stf_data = skutil.view_as_blocks(
        data_[:n_sample], (n_stft_sample, n_channel)
    ).squeeze(
        axis=1
    )  # (n_stf, N_stft_sample, n_channel)

    window = windows.tukey(M=n_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (n_stf, N_stft_sample, n_channel)
    n_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (n_stf, N_stft_sample, n_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * n_stft_sample / rate_)
    idx_end = int((fc + 0.5 * bw) * n_stft_sample / rate_)
    collapsed_spectrum = np.sum(stft_data[:, idx_start : idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    return collapsed_spectrum.reshape(n_stf, -1, 1).conj() * collapsed_spectrum.reshape(
        n_stf, 1, -1
    )


# noinspection PyArgumentList
def eigh_max(a: np.ndarray) -> custom_types.Numeric:
    r"""
    Evaluate :math:`\mu_{\max}(\bbB)` with

    :math:
    B = (\overline{\bbA} \circ \bbA)^{H} (\overline{\bbA} \circ \bbA)
    """
    if a.ndim != 2:
        raise ValueError(
            f"`a` has wrong dimensions (expected ndim == 2, got ndim {a.ndim})."
        )

    def matvec(v: np.ndarray) -> custom_types.Numeric:
        v = v.reshape(-1)
        c = (a * v) @ a.conj().T
        d = c @ a
        return np.sum(a.conj() * d, axis=0).real

    m, n = a.shape
    b = splinalg.LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)
    d_max = splinalg.eigsh(b, k=1, which="LM", return_eigenvectors=False)
    return d_max[0]


# noinspection PyUnresolvedReferences
def _solve(
    functions: list,
    x0: np.ndarray,
    solver: Callable,
    atol: Optional[custom_types.Numeric] = None,
    dtol: Optional[custom_types.Numeric] = None,
    rtol: Optional[custom_types.Numeric] = 1e-3,
    xtol: Optional[custom_types.Numeric] = None,
    maxit: Optional[custom_types.Numeric] = 200,
) -> dict:
    """
    Solve an optimization problem whose objective function is the sum of some
    convex functions.
    """
    # Add a second dummy convex function if only one function is provided.
    if len(functions) < 1:
        raise ValueError("At least 1 convex function should be provided.")
    elif len(functions) == 1:
        functions.append(dummy())

    # Set solver and functions verbosity.
    solver.verbosity = "NONE"
    for f in functions:
        f.verbosity = "NONE"

    tstart = time.time()
    crit = None
    niter = 0
    objective = [[f.eval(x0) for f in functions]]
    rtol_only_zeros = True

    # Solver specific initialization.
    solver.pre(functions, x0)
    tape_buffer = np.zeros((1000, len(x0)))
    tape_buffer[0] = x0

    while not crit:
        niter += 1

        if xtol is not None:
            last_sol = np.array(solver.sol, copy=True)

        # Solver iterative algorithm.
        solver.algo(objective, niter)
        tape_buffer[niter] = solver.sol

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Verify stopping criteria.
        if atol is not None and current < atol:
            crit = "ATOL"
        if dtol is not None and np.abs(current - last) < dtol:
            crit = "DTOL"
        if rtol is not None:
            div = current  # Prevent division by 0.
            if div == 0:
                if last != 0:
                    div = last
                else:
                    div = 1.0  # Result will be zero anyway.
            else:
                rtol_only_zeros = False
            relative = np.abs((current - last) / div)
            if relative < rtol and not rtol_only_zeros:
                crit = "RTOL"
        if xtol is not None:
            err = np.linalg.norm(solver.sol - last_sol)  # noqa
            err /= np.sqrt(last_sol.size)
            if err < xtol:
                crit = "XTOL"
        if maxit is not None and niter >= maxit:
            crit = "MAXIT"

    # Returned dictionary.
    result = {
        "sol": solver.sol,
        "solver": solver.__class__.__name__,  # algo for consistency ?
        "crit": crit,
        "niter": niter,
        "time": time.time() - tstart,
        "objective": objective,
    }
    try:
        # Update dictionary for primal-dual solvers
        result["dual_sol"] = solver.dual_sol
    except AttributeError:
        pass

    # Solver specific post-processing (e.g. delete references).
    solver.post()

    result["backtrace"] = tape_buffer[: (niter + 1)]
    return result


def solve(
    s: np.ndarray,
    a: np.ndarray,
    lambda_: Optional[custom_types.Numeric] = None,
    gamma: Optional[custom_types.Numeric] = 0.5,
    l_: Optional[custom_types.Numeric] = None,
    d: Optional[custom_types.Numeric] = 50,
    x0: Optional[np.ndarray] = None,
    eps: Optional[custom_types.Numeric] = 1e-3,
    n_iter_max: Optional[custom_types.Numeric] = 200,
    momentum: Optional[bool] = True,
) -> dict:
    """
    APGD solution to the Acoustic Camera problem.
    """
    m, n = a.shape
    if not ((s.shape[0] == s.shape[1]) and (s.shape[0] == m)):
        raise ValueError("Parameters `s` and `a` are inconsistent.")
    if not np.allclose(s, s.conj().T):
        raise ValueError("Parameter `s` must be Hermitian.")

    if not (0 <= gamma <= 1):
        raise ValueError("Parameter `gamma` is must lie in [0, 1].")

    if l_ is None:
        l_ = 2 * eigh_max(a)
    elif l_ <= 0:
        raise ValueError("Parameter `l_` must be positive.")

    if d < 2:
        raise ValueError(r"Parameter[d] must be \ge 2.")

    if x0 is None:
        x0 = np.zeros((n,), dtype=np.float64)
    elif np.any(x0 < 0):
        raise ValueError("Parameter `x0` must be non-negative.")

    if not (0 < eps < 1):
        raise ValueError("Parameter `eps` must lie in (0, 1).")

    if n_iter_max < 1:
        raise ValueError("Parameter `N_iter_max` must be positive.")

    if lambda_ is None:
        if gamma > 0:  # Procedure of Remark 3.4
            # When gamma == 0, we fall into the ridge-regularizer case, so no
            # need to do the following.
            func = [L2Loss(s, a), ElasticNetLoss(lambda_=0, gamma=gamma)]
            solver = opt.solvers.forward_backward(
                accel=GroundTruthAccel(d, l_, momentum=False)
            )
            i_opt = _solve(
                functions=func,
                x0=np.zeros((n,)),
                solver=solver,
                rtol=eps,
                maxit=1,
            )
            alpha = 1 / l_
            lambda_ = np.max(i_opt["sol"]) / (10 * alpha * gamma)
        else:
            lambda_ = 1  # Anything will do.
    elif lambda_ < 0:
        raise ValueError("Parameter `lambda_` must be non-negative.")

    func = [L2Loss(s, a), ElasticNetLoss(lambda_, gamma)]
    solver = opt.solvers.forward_backward(accel=GroundTruthAccel(d, l_, momentum))
    i_opt = _solve(
        functions=func,
        x0=x0.copy(),
        solver=solver,
        rtol=eps,
        maxit=n_iter_max,
    )
    i_opt["gamma"] = gamma
    i_opt["lambda_"] = lambda_
    i_opt["L"] = l_
    return i_opt


def form_visibility(
    data: np.ndarray,
    rate: custom_types.Numeric,
    fc: custom_types.Numeric,
    bw: custom_types.Numeric,
    t_sti: custom_types.Numeric,
    t_stationarity: custom_types.Numeric,
) -> np.ndarray:
    """
    Compute visibilities in the frequency domain
    """
    s_sti = extract_visibilities(data, rate, t_sti, fc, bw, alpha=1.0)
    n_sample, n_channel = data.shape
    n_sti_per_stationary_block = int(t_stationarity / t_sti)
    return (
        skutil.view_as_windows(
            s_sti,
            window_shape=(n_sti_per_stationary_block, n_channel, n_channel),
            step=(n_sti_per_stationary_block, n_channel, n_channel),
        )
        .squeeze(axis=(1, 2))
        .sum(axis=1)
    )


def get_visibility_matrix(
    audio_in: np.ndarray,
    micarray_coords: np.ndarray,
    sr: Optional[custom_types.Numeric] = config.SAMPLE_RATE,
    t_sti: Optional[custom_types.Numeric] = config.AIMG_TSTI,
    scale: Optional[str] = config.AIMG_SCALE,
    nbands: Optional[custom_types.Numeric] = config.AIMG_NBANDS,
    frame_cap: Optional[custom_types.Numeric] = config.AIMG_FRAME_CAP,
    fmin: Optional[custom_types.Numeric] = config.AIMG_FMIN,
    fmax: Optional[custom_types.Numeric] = config.AIMG_FMAX,
    bw: Optional[custom_types.Numeric] = config.AIMG_BANDWIDTH,
    sh_order: Optional[custom_types.Numeric] = config.AIMG_SH_ORDER,
) -> np.ndarray:
    """
    Compute visibility matrix from audio data using accelerated proximal gradient descent (APGD) algorithm.

    Arguments:
        audio_in (np.ndarray): audio matrix with shape (samples, channels)
        micarray_coords (np.ndarray): polar coordinates of micarray capsules in form (azimuth, elevation, distance).
            Must have shape (capsules, 3), where n_capsules == n_channels of audio
        sr (Numeric): sample rate of the audio
        t_sti (Numeric): frame length, defaults to 100 ms (same as DCASE label resolution)
        scale (str): scaling to use for `nbands` frequency bands, must be either "linear" or "log"
        nbands (Numeric): number of frequency bands
        frame_cap (Numeric): maximum number of frames to compute: set to `None` to use all frames
        fmin (Numeric): minimum frequency for `nbands` frequency bands
        fmax (Numeric): maximum frequency for `nbands` frequency bands
        bw (Numeric): bandwidth for `nbands` frequency bands
        sh_order (Numeric): spherical harmonic order that determines sampling density: higher values make denser fields

    Returns:
        np.ndarray: the acoustic image with shape (tesselation, bands, frames)
    """
    # frequency bands
    if fmin >= fmax:
        raise ValueError(
            f"Minimum frequency must be smaller than maximum frequency "
            f"(current minimum: {fmin}, maximum: {fmax})."
        )
    if scale == "linear":
        freq = np.linspace(fmin, fmax, nbands)
    elif scale == "log":
        freq = librosa.mel_frequencies(n_mels=nbands, fmin=fmin, fmax=fmax)
    else:
        raise ValueError(
            f"'{scale}' is not a valid scale to generate covariance matrices "
            f"(must be either 'log' or 'linear')"
        )

    # spherical field
    r = get_field(sh_order)
    micarray_coords_conv = _convert_mic_coordinates(micarray_coords)
    xyz = get_mic_xyz_coords(micarray_coords_conv)
    dev_xyz = np.array(xyz).T

    # steering operator
    a = steering_operator(dev_xyz, r)
    n_px = a.shape[1]
    apgd_map = []

    # Process bands
    for i in range(nbands):

        # somehow, need to do this
        t_stationarity = 10 * t_sti
        s = form_visibility(audio_in, sr, freq[i], bw, t_sti, t_stationarity)
        n_sample = s.shape[0]

        # Cap frames if required
        if frame_cap:
            s = s[:frame_cap, :, :]
            n_sample = frame_cap

        visibilities_per_frame = []
        apgd_gamma = 0.5
        apgd_per_band = np.zeros((n_sample, n_px))
        i_prev = np.zeros((n_px,))

        for s_idx in tqdm(range(n_sample), desc=f"Computing band {i}..."):

            # Eigen-decomposition
            s_d, s_v = linalg.eigh(s[s_idx])

            # Clamp results
            if s_d.max() <= 0:
                s_d[:] = 0
            else:
                s_d = np.clip(s_d / s_d.max(), 0, None)
            s_norm = (s_v * s_d) @ s_v.conj().T

            visibilities_per_frame.append(s_norm)

            # gradient descent
            i_apgd = solve(s_norm, a, gamma=apgd_gamma, x0=i_prev.copy())
            apgd_per_band[s_idx] = i_apgd["sol"]
            i_prev = i_apgd["sol"]

        apgd_map.append(apgd_per_band)

    # (tesselation, bands, frames)
    apgd_arr = np.array(apgd_map)
    return apgd_arr.transpose((2, 0, 1))


def create_target_grid(
    width: custom_types.Numeric, height: custom_types.Numeric
) -> np.ndarray:
    """
    Create regular target grid of points based on given width and height
    """
    target_az = np.linspace(
        180, -180, utils.sanitise_positive_number(width, cast_to=int)
    )
    target_el = np.linspace(
        90, -90, utils.sanitise_positive_number(height, cast_to=int)
    )
    target_az_grid, target_el_grid = np.meshgrid(target_az, target_el, indexing="xy")
    return np.stack([target_az_grid.ravel(), target_el_grid.ravel()], axis=1)


def create_2d_gaussian(
    cx: custom_types.Numeric,
    cy: custom_types.Numeric,
    width: custom_types.Numeric,
    height: custom_types.Numeric,
    circle_radius: custom_types.Numeric = config.AIMG_CIRCLE_RADIUS_DEG,
) -> np.ndarray:
    """
    Compute a 2D Gaussian centered at `cx, cy` pixels.

    The radius of the circle is set to contain 2 SD of the values within the span of (width, height)
    """
    # Check inputs are valid
    if not 0 <= cx <= width:
        raise ValueError(
            f"X coordinate is outside of width! (x = {cx}, width = {width})"
        )
    if not 0 <= cy <= height:
        raise ValueError(
            f"Y coordinate is outside of height! (y = {cy}, height = {height})"
        )

    # The circle should contain 2 SD of the vals (68-*95*-99.7% rule)
    sigma_deg = circle_radius / 2.0

    deg_per_pixel_x = 360.0 / width
    deg_per_pixel_y = 180.0 / height

    _, center_elevation_deg = _equirectangular_to_spherical(
        cx, cy, width=width, height=height
    )

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (H, W)

    # Wrapped pixel deltas (preserve sign)
    dx = (xx - cx + width / 2) % width - width / 2
    dy = yy - cy

    # Convert to angular deltas
    delta_az_deg = -dx * deg_per_pixel_x  # azimuth increases leftward
    delta_el_deg = dy * deg_per_pixel_y

    cos_lat = np.cos(np.radians(center_elevation_deg))

    dist_sq_deg = (delta_el_deg**2) + (cos_lat * delta_az_deg) ** 2

    gaussian = np.exp(-dist_sq_deg / (2.0 * sigma_deg**2))

    return gaussian


def find_contours(acoustic_image: np.ndarray) -> list[np.ndarray]:
    """
    Find contours in an equirectangular image. Horizontal wrap-around is handled naturally:
      - If a blob is split across left/right edges, findContours returns two separate contours
      - Both contours are kept in the segmentation list

    Args:
        acoustic_image (np.ndarray): 2D acoustic image (already scaled/masked)

    Returns:
        list[np.ndarray]: list of contours
    """
    cv2 = utils.safe_import("cv2")

    # Binary mask
    binary_mask = (acoustic_image > 0).astype(np.uint8) * 255

    # Find contours normally
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # [(n_coordinates, 2), (n_coordinates, 2), ...]
    #  this will be len == 1 in all cases BUT when a sound event wraps around both edges of an image
    return [c.squeeze() for c in contours]


def get_segmentation_pixels(
    acoustic_image: np.ndarray, contour_boundary: np.ndarray
) -> list[list]:
    """
    Given an acoustic image and contour, compute pixel coordinate values of contour and return list of lists with
    inner structure [x_coord, y_coord, amplitude]
    """
    # Python already caches imports so this should be fairly quick to do again
    cv2 = utils.safe_import("cv2")

    # We can just grab height and width from the acoustic image directly
    height, width = acoustic_image.shape

    # Compute the mask and fill with the contour boundary
    mask__ = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask__, [contour_boundary.astype(np.int32)], 255)

    # Stack to get (x, y, amplitude), with shape (N_coordinates, 3)
    y_coords, x_coords = np.where(mask__ == 255)
    amplitude_values = acoustic_image[y_coords, x_coords]
    pixels_data = np.column_stack([x_coords, y_coords, amplitude_values])

    # Return as a list of [x_coord, y_coord, amplitude] lists
    return [[int(x), int(y), amp] for (x, y, amp) in pixels_data.tolist()]


def generate_acoustic_image_json(
    acoustic_image: np.ndarray,
    metadata: np.ndarray,
    resolution: Optional[
        tuple[custom_types.Numeric, custom_types.Numeric]
    ] = config.AIMG_RESOLUTION,
    polygon_mask_threshold: Optional[
        custom_types.Numeric
    ] = config.AIMG_POLYGON_MASK_THRESHOLD,
    circle_radius: Optional[custom_types.Numeric] = config.AIMG_CIRCLE_RADIUS_DEG,
) -> list[dict]:
    """
    Generates a list of dictionaries (JSON-style) for a given acoustic image.

    The function presupposes both an acoustic image with shape (tesselation, bands, frames) and an array of metadata
    (computed using `audiblelight.synthesize.generate_dcase2024_metadata`, or similar). The method used is as follows:
        1. Take the median energy for each band in the acoustic image: gives (tesselation, frames)
        2. Iterate over all frames with an active annotation in the metadata array
            2a. Interpolate the corresponding acoustic image frame to an image with shape (height, width)
            2b. Iterate over all annotations for the current frame:
                2bi. Create a 2D Gaussian centered at the X and Y pixel coordinates of the annotation, with radius set
                        to span 2SD of all pixel values
                2bii. Scale the acoustic image frame by multiplying by the Gaussian
                2biii. Mask all values in the scaled acoustic image frame that are below `polygon_mask_threshold`
                2biv. Apply contour detection to grab the edges of each "blob" in the image
            2c. Append all "blobs" for the frame: each of these have the format [x_pixel, y_pixel, amplitude]
        3. Return a full dictionary containing annotations of every frame

    The dictionaries contain the following keys:
        - "metadata_frame_index": the index of the frame within the acoustic image
        - "instance_id": a unique integer identifier for each event in the scene
        - "category_id": the index of the soundevent
        - "distance": the distance of the soundevent
        - "segmentation": a list of [x_pixel, y_pixel, amplitude] values for every segmentation in that frame.

    Note that all but the last value are taken directly from the metadata: only "segemntation" is defined by the
    acoustic image.

    Finally, it is also assumed that the amplitude values should be scaled *across* multiple JSON files that constitute
    an entire dataset, e.g. by Z-scoring, scaling between 0 and 1, etc. As this process relies on summary statistics
    that cannot easily be known when computing individual JSONs, this must be accomplished after calling this function.

    Arguments:
        acoustic_image (np.ndarray): an acoustic image with shape (tesselation, bands, frames)
        metadata (np.ndarray): an array of metadata corresponding to the acoustic image
        resolution (tuple): the resolution to interpolate the image to: must be equirectangular, in form (width, height)
        polygon_mask_threshold (Numeric): after scaling the acoustic image according to the 2D Gaussian, values below
            this threshold will be set to 0. This value should be tweaked based on looking at the shape of the images.
        circle_radius (Numeric): the radius of the circle placed at ground-truth azimuth and elevation points when
            calculating the 2D Gaussian

    Returns:
        list[dict]: the metadata extracted for this acoustic image
    """
    # Validate the acoustic image
    if not acoustic_image.ndim == 3:
        raise ValueError(
            f"Expected acoustic image to have 3 dimensions, but got {acoustic_image.shape}"
        )

    # Store scene-wide results here
    scene_res = []

    # Unpack acoustic image
    n_tesselation, n_bands, n_frames = acoustic_image.shape

    # Compute median over bands once for entire acoustic image: shape (tesselation, frames)
    acoustic_image_medianed = np.median(acoustic_image, axis=1)

    # We can infer the `sh_order` used directly from the acoustic image, there's no need to pass it as an argument
    sh_order = int(math.sqrt(n_tesselation) / 2 - 1)

    # Get the tesselation coordinates and convert to spherical: shape (n_px, 2)
    tesselation = fibonacci(sh_order).T
    tesselation_eq = np.apply_along_axis(
        lambda x: _cartesian_to_spherical(*x), 1, tesselation
    )

    # Unpack video resolution
    video_width, video_height = resolution

    # Create regular target grid based on (scaled) width and height
    target_points = create_target_grid(video_width, video_height)

    # Grab frames with ground truth annotations and iterate over these
    frames_with_gt_annotations = np.unique(metadata[:, 0])
    for metadata_frame_idx in frames_with_gt_annotations:

        # Grab the corresponding acoustic image frame
        acoustic_image_frame = acoustic_image_medianed[:, metadata_frame_idx]

        # Interpolate the acoustic image for this frame and reshape to (height, width)
        acoustic_image_interpolated = griddata(
            tesselation_eq,
            acoustic_image_frame,
            target_points,
            method="linear",
            fill_value=0.0,
        ).reshape(video_height, video_width)

        # Grab the annotations for this frame and iterate over
        #  We can have multiple annotations per frame, so this will be an array with min len == 1
        current_frame_metadatas = metadata[metadata[:, 0] == metadata_frame_idx]
        for metadata_row in current_frame_metadatas:

            # Grab everything from the row of metadata
            _, class_id, instance_id, gt_az, gt_el, gt_dist = metadata_row[:6]

            # Convert spherical azimuth/elevation to equirectangular
            gt_az_eq, gt_el_eq = _spherical_to_equirectangular(
                gt_az, gt_el, width=video_width, height=video_height
            )

            # Compute the 2D Gaussian centered at (azimuth, elevation): shape (width, height)
            gauss_gt = create_2d_gaussian(
                gt_az_eq,
                gt_el_eq,
                width=video_width,
                height=video_height,
                circle_radius=circle_radius,
            )

            # Multiply the acoustic image by the Gaussian to scale it
            acoustic_image_gauss_scaled = acoustic_image_interpolated * gauss_gt

            # Mask values in the scaled image that are below the threshold
            acoustic_image_gauss_masked = acoustic_image_gauss_scaled.copy()
            polygon_mask = np.where(
                acoustic_image_gauss_masked < polygon_mask_threshold
            )
            acoustic_image_gauss_masked[polygon_mask] = 0

            # Find contours within the masked image
            contours = find_contours(acoustic_image_gauss_masked)

            # We'll store segmentations for this frame inside here
            segmentations = []

            # Iterate over all the contours we've found
            for contour in contours:

                # skip degenerate contours
                if contour.ndim == 1:
                    continue

                # Grab the pixels + amplitude values within this segmentation and append to the list
                pixels_list = get_segmentation_pixels(
                    acoustic_image_gauss_masked, contour
                )
                segmentations.append(pixels_list)

            # Now we can create the annotations dictionary
            annotations_dict = {
                "metadata_frame_index": int(metadata_frame_idx),
                "instance_id": int(instance_id),
                "category_id": int(class_id),
                "segmentation": segmentations,
                "distance": float(gt_dist),
            }
            scene_res.append(annotations_dict)

    return scene_res


def standardise_acoustic_image_amplitude(
    acoustic_image_labels: list[dict],
) -> list[dict]:
    """
    Standardise the amplitude values within the acoustic image labels using distribution from STARSS23 training data.

    The process is as follows:
        - We start by grabbing the mean and standard deviation amplitude values within the STARSS23 training set
            - Note that these values are HARDCODED as they have been found from empirical testing with this data
        - Each dictionary in the input consists of 1/2 polygons for a single object in a single frame
            - Usually, we'd expect to just have one polygon if the object is in the middle of the frame, but if it
                crosses to the edge of the frame, it will wrap around, so we'll end up with two polygons.
        - For each polygon, take the amplitude values, subtract STARSS_MU, and divide by STARSS_SD (i.e., z-scoring)
            - Next, add 0.5 to the Z-scored values and clip anything above 1 and below 0.01

    This process then standardises the amplitude values of the synthetic data generated with AudibleLight to lie
    within the range of the (real) data contained inside the STARSS23 training data.
    """

    # These values are hardcoded, should not be changed
    starss23_mu, starss23_sigma = config.AIMG_STARSS23_MU, config.AIMG_STARSS23_SIGMA

    # Store standardised results
    res_std = []

    # Iterate over all the labels
    for aimg in acoustic_image_labels:

        new_polys = []

        # Grab the polygons for this label and iterate over them
        for poly in aimg["segmentation"]:
            poly_arr = np.array(poly)
            poly_amp = poly_arr[:, -1]

            # Z-score
            poly_amp = (poly_amp - starss23_mu) / starss23_sigma

            # Add 0.5 and clip
            poly_amp = np.clip(poly_amp + 0.5, 0.01, 1.0)

            # Create new array and replace the amplitude values with standardised version
            poly_new = poly_arr.copy()
            poly_new[:, -1] = poly_amp

            new_polys.append(poly_new.tolist())

        # Update the list with the new polygons and append everything
        aimg["segmentation"] = new_polys
        res_std.append(aimg)

    # Return standardised results
    return res_std
