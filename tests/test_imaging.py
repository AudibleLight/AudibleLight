#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test cases for functionality inside audiblelight/imaging.py"""

import math

import numpy as np
import pytest

# noinspection PyProtectedMember
from audiblelight.imaging import (
    ElasticNetLoss,
    GroundTruthAccel,
    L2Loss,
    _cartesian_to_spherical,
    _convert_mic_coordinates,
    _degrees_to_radians,
    _equirectangular_to_cartesian,
    _equirectangular_to_spherical,
    _polar_to_cartesian,
    _spherical_to_equirectangular,
    create_2d_gaussian,
    create_target_grid,
    eigh_max,
    fibonacci,
    find_contours,
    generate_acoustic_image_json,
    get_field,
    get_mic_xyz_coords,
    get_segmentation_pixels,
    get_visibility_matrix,
    solve,
    standardise_acoustic_image_amplitude,
)
from audiblelight.micarrays import Eigenmike32
from audiblelight.utils import SMALL


@pytest.mark.parametrize(
    "old_coords,expected_coords",
    [
        (
            Eigenmike32().coordinates_polar,
            {
                "1": [69, 0, 0.042],
                "2": [90, 32, 0.042],
                "3": [111, 0, 0.042],
                "4": [90, 328, 0.042],
                "5": [32, 0, 0.042],
                "6": [55, 45, 0.042],
                "7": [90, 69, 0.042],
                "8": [125, 45, 0.042],
                "9": [148, 0, 0.042],
                "10": [125, 315, 0.042],
                "11": [90, 291, 0.042],
                "12": [55, 315, 0.042],
                "13": [21, 91, 0.042],
                "14": [58, 90, 0.042],
                "15": [121, 90, 0.042],
                "16": [159, 89, 0.042],
                "17": [69, 180, 0.042],
                "18": [90, 212, 0.042],
                "19": [111, 180, 0.042],
                "20": [90, 148, 0.042],
                "21": [32, 180, 0.042],
                "22": [55, 225, 0.042],
                "23": [90, 249, 0.042],
                "24": [125, 225, 0.042],
                "25": [148, 180, 0.042],
                "26": [125, 135, 0.042],
                "27": [90, 111, 0.042],
                "28": [55, 135, 0.042],
                "29": [21, 269, 0.042],
                "30": [58, 270, 0.042],
                "31": [122, 270, 0.042],
                "32": [159, 271, 0.042],
            },
        )
    ],
)
def test_convert_mic_coordinates(old_coords, expected_coords):
    from deepdiff import DeepDiff

    out = _convert_mic_coordinates(old_coords)
    dd = DeepDiff(out, expected_coords)
    assert len(dd) == 0


@pytest.mark.parametrize(
    "sh_order,expected_shape",
    [
        (10, (3, 484)),
    ],
)
def test_get_field(sh_order, expected_shape):
    out = get_field(sh_order)
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "coords_dict,expected",
    [
        # Single microphone at origin
        ({"mic1": [0, 0, 1]}, {"mic1": [0, 0, 1]}),
        # 90 degree angles
        ({"mic1": [90, 90, 1]}, {"mic1": [math.pi / 2, math.pi / 2, 1]}),
        # 180 degree angles
        ({"mic1": [180, 180, 5]}, {"mic1": [math.pi, math.pi, 5]}),
        # Multiple microphones
        (
            {"mic1": [0, 0, 1], "mic2": [45, 90, 2], "mic3": [90, 180, 3]},
            {
                "mic1": [0, 0, 1],
                "mic2": [math.pi / 4, math.pi / 2, 2],
                "mic3": [math.pi / 2, math.pi, 3],
            },
        ),
        # Negative angles
        ({"mic1": [-45, -90, 1]}, {"mic1": [math.radians(-45), math.radians(-90), 1]}),
        # Full rotation
        ({"mic1": [360, 360, 1]}, {"mic1": [2 * math.pi, 2 * math.pi, 1]}),
        # Empty dict
        ({}, {}),
    ],
)
def test_degrees_to_radians(coords_dict, expected):
    result = _degrees_to_radians(coords_dict)
    assert len(result) == len(expected)
    for key in expected:
        assert key in result
        for i in range(3):
            assert math.isclose(result[key][i], expected[key][i], rel_tol=SMALL)


@pytest.mark.parametrize(
    "coords_dict,units,expected",
    [
        # Point along z-axis (colatitude=0)
        ({"mic1": [0, 0, 1]}, "radians", {"mic1": [0, 0, 1]}),
        # Point in xy-plane (colatitude=90°) at azimuth=0°
        ({"mic1": [90, 0, 1]}, "degrees", {"mic1": [1, 0, 0]}),
        # Point in xy-plane at 90° azimuth
        ({"mic1": [90, 90, 1]}, "degrees", {"mic1": [0, 1, 0]}),
        # Point at 45° colatitude, 0° azimuth
        (
            {"mic1": [45, 0, 1]},
            "degrees",
            {"mic1": [math.sin(math.radians(45)), 0, math.cos(math.radians(45))]},
        ),
        # Multiple microphones with different radii
        (
            {"mic1": [math.pi / 2, 0, 2], "mic2": [math.pi / 2, math.pi / 2, 3]},
            "radians",
            {"mic1": [2, 0, 0], "mic2": [0, 3, 0]},
        ),
        # Point at 180° colatitude (negative z)
        ({"mic1": [180, 0, 1]}, "Degrees", {"mic1": [0, 0, -1]}),  # lower case works
    ],
)
def test_polar_to_cartesian(coords_dict, units, expected):
    result = _polar_to_cartesian(coords_dict, units)
    assert len(result) == len(expected)
    for key in expected:
        assert key in result
        for i in range(3):
            assert math.isclose(
                result[key][i], expected[key][i], rel_tol=SMALL, abs_tol=SMALL
            )


@pytest.mark.parametrize(
    "mic_coords, expected", [({"mic1": [180, 0, 1]}, [[0, 0, -1]])]
)
def test_get_mic_xyz_coords(mic_coords, expected):
    out = get_mic_xyz_coords(mic_coords)
    assert np.allclose(out, expected, atol=SMALL)


@pytest.mark.parametrize(
    "n, direction, fo_v, match",
    [
        (-1, None, None, "Parameter `n` must be non-negative."),  # n < 0
        (
            1,
            [1, 0, 0],
            None,
            "Parameter `fo_v` must be specified if `direction` is provided.",
        ),  # direction w/o fo_v
        (
            1,
            [1, 0, 0],
            np.deg2rad(400),
            "Parameter `fo_v` must be in \\(0, 360\\) degrees.",
        ),  # fo_v out of range
    ],
)
def test_fibonacci_invalid_inputs(n, direction, fo_v, match):
    with pytest.raises(ValueError, match=match):
        _ = fibonacci(n, direction=direction, fo_v=fo_v)


@pytest.mark.parametrize(
    "n, direction, fo_v",
    [
        (2, None, None),  # full sphere
        (2, [0, 0, 1], np.deg2rad(90)),  # region-limited
    ],
)
def test_fibonacci_valid_generation(n, direction, fo_v):
    xyz = fibonacci(n, direction=direction, fo_v=fo_v)
    assert xyz.shape[0] == 3, "Output must have 3 rows (x, y, z)."
    assert np.all(
        np.linalg.norm(xyz, axis=0) - 1 < 1e-12
    ), "All points must lie on the unit sphere."

    if direction is not None:
        direction = np.array(direction) / np.linalg.norm(direction)
        cos_theta = direction @ xyz
        assert np.all(
            cos_theta >= np.cos(fo_v / 2)
        ), "All points must lie within specified FOV."


def test_get_visibility_matrix_invalid_inputs():
    with pytest.raises(
        ValueError, match="Minimum frequency must be smaller than maximum frequency"
    ):
        _ = get_visibility_matrix(
            audio_in=np.zeros((32, 44100)),
            micarray_coords=np.zeros((32, 3)),
            fmin=4500,
            fmax=1500,
        )

    with pytest.raises(ValueError, match="'asdf' is not a valid scale"):
        _ = get_visibility_matrix(
            audio_in=np.zeros((32, 44100)),
            micarray_coords=np.zeros((32, 3)),
            scale="asdf",
        )


@pytest.mark.parametrize(
    "coords_dict,units,error_type",
    [
        ({"mic1": [0, 0, 1]}, None, ValueError),
        ({"mic1": [0, 0, 1]}, 123, ValueError),
        ({"mic1": [0, 0, 1]}, "invalid", ValueError),
    ],
)
def test_polar_to_cartesian_invalid_units(coords_dict, units, error_type):
    with pytest.raises(error_type):
        _polar_to_cartesian(coords_dict, units)


@pytest.mark.parametrize(
    "x,y,z,expected_azimuth,expected_elevation",
    [
        # Point along positive x-axis
        (1, 0, 0, 0, 0),
        # Point along positive y-axis
        (0, 1, 0, 90, 0),
        # Point along negative x-axis
        (-1, 0, 0, 180, 0),
        # Point along negative y-axis
        (0, -1, 0, -90, 0),
        # Point along positive z-axis (unit sphere)
        (0, 0, 1, 0, 90),
        # Point along negative z-axis (unit sphere)
        (0, 0, -1, 0, -90),
        # 45° elevation, 45° azimuth
        (
            np.cos(np.radians(45)) * np.cos(np.radians(45)),
            np.cos(np.radians(45)) * np.sin(np.radians(45)),
            np.sin(np.radians(45)),
            45,
            45,
        ),
        # Arrays
        (
            np.array([1, 0, -1]),
            np.array([0, 1, 0]),
            np.array([0, 0, 0]),
            np.array([0, 90, 180]),
            np.array([0, 0, 0]),
        ),
    ],
)
def test_cartesian_to_spherical(x, y, z, expected_azimuth, expected_elevation):
    azimuth, elevation = _cartesian_to_spherical(x, y, z)

    if isinstance(expected_azimuth, np.ndarray):
        np.testing.assert_allclose(azimuth, expected_azimuth, rtol=SMALL, atol=SMALL)
        np.testing.assert_allclose(
            elevation, expected_elevation, rtol=SMALL, atol=SMALL
        )
    else:
        assert math.isclose(azimuth, expected_azimuth, rel_tol=SMALL, abs_tol=SMALL)
        assert math.isclose(elevation, expected_elevation, rel_tol=SMALL, abs_tol=SMALL)


@pytest.mark.parametrize(
    "azimuth_deg,elevation_deg,width,height,expected_x,expected_y",
    [
        # Center of image (azimuth=0, elevation=0)
        (0, 0, 360, 180, 180, 90),
        # Top center (elevation=90)
        (0, 90, 360, 180, 180, 0),
        # Bottom center (elevation=-90)
        (0, -90, 360, 180, 180, 180),
        # Left edge (azimuth=-180)
        (-180, 0, 360, 180, 0, 90),
        # Right edge (azimuth=180)
        (180, 0, 360, 180, 0, 90),  # Wraps around
        # Different image size
        (0, 0, 1920, 1080, 960, 540),
        # Positive azimuth
        (90, 0, 360, 180, 90, 90),
        # Negative azimuth
        (-90, 0, 360, 180, 270, 90),
        # Top right quadrant
        (45, 45, 360, 180, 135, 45),
    ],
)
def test_spherical_to_equirectangular(
    azimuth_deg, elevation_deg, width, height, expected_x, expected_y
):
    x, y = _spherical_to_equirectangular(azimuth_deg, elevation_deg, width, height)
    assert x == expected_x
    assert y == expected_y


@pytest.mark.parametrize(
    "x,y,width,height,expected_azimuth,expected_elevation",
    [
        # Center of image
        (180, 90, 360, 180, 0, 0),
        # Top left corner
        (0, 0, 360, 180, 180, 90),
        # Bottom right corner
        (360, 180, 360, 180, -180, -90),
        # Top center
        (180, 0, 360, 180, 0, 90),
        # Bottom center
        (180, 180, 360, 180, 0, -90),
        # Different image size
        (960, 540, 1920, 1080, 0, 0),
        # Left edge center
        (0, 90, 360, 180, 180, 0),
        # Right edge center
        (360, 90, 360, 180, -180, 0),
        # Quarter point
        (90, 45, 360, 180, 90, 45),
    ],
)
def test_equirectangular_to_spherical(
    x, y, width, height, expected_azimuth, expected_elevation
):
    azimuth, elevation = _equirectangular_to_spherical(x, y, width, height)
    assert math.isclose(azimuth, expected_azimuth, rel_tol=SMALL, abs_tol=SMALL)
    assert math.isclose(elevation, expected_elevation, rel_tol=SMALL, abs_tol=SMALL)


@pytest.mark.parametrize(
    "azimuth_orig, elevation_orig",
    [
        (0, 0),
        (45, 30),
        (-90, -45),
        (135, 60),
        (-45, -30),
    ],
)
def test_spherical_equirectangular_round_trip(azimuth_orig, elevation_orig):
    """
    Test that converting spherical->equirectangular->spherical gives original values
    """
    width, height = 360, 180

    x, y = _spherical_to_equirectangular(azimuth_orig, elevation_orig, width, height)
    azimuth_new, elevation_new = _equirectangular_to_spherical(x, y, width, height)

    assert math.isclose(azimuth_orig, azimuth_new, rel_tol=SMALL, abs_tol=SMALL)
    assert math.isclose(elevation_orig, elevation_new, rel_tol=SMALL, abs_tol=SMALL)


@pytest.mark.parametrize("width, height", [(900, 450), (450, 225), (1800, 900)])
def test_create_target_grid(width, height):
    out = create_target_grid(width, height)
    assert out.shape == (width * height, 2)
    assert out[:, 0].max() == 180
    assert out[:, 0].min() == -180
    assert out[:, 1].max() == 90
    assert out[:, 1].min() == -90


@pytest.mark.parametrize(
    "r,lat,lon,expected_x,expected_y,expected_z",
    [
        # Point at equator, prime meridian
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        # Point at north pole
        (1.0, np.pi / 2, 0.0, 0.0, 0.0, 1.0),
        # Point at south pole
        (1.0, -np.pi / 2, 0.0, 0.0, 0.0, -1.0),
        # Point at equator, 90° east
        (1.0, 0.0, np.pi / 2, 0.0, 1.0, 0.0),
        # Radius = 2 at equator
        (2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
        # Zero radius (origin)
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_equirectangular_to_cartesian(r, lat, lon, expected_x, expected_y, expected_z):
    """Test equirectangular to cartesian coordinate conversion"""
    result = _equirectangular_to_cartesian(r, lat, lon)
    x, y, z = result[:, 0]

    assert math.isclose(x, expected_x, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(y, expected_y, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(z, expected_z, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize(
    "r,lat,lon",
    [
        (-1.0, 0.0, 0.0),
        (-100.0, np.pi / 4, np.pi / 4),
        (np.array([-1.0, -2.0]), 0.0, 0.0),
    ],
)
def test_equirectangular_to_cartesian_negative_radius(r, lat, lon):
    """Test that negative radius raises ValueError"""
    with pytest.raises(ValueError, match="must be non-negative"):
        _equirectangular_to_cartesian(r, lat, lon)


@pytest.mark.parametrize(
    "cx,cy,width,height", [(450, 450, 900, 900), (850, 50, 860, 60)]
)
def test_create_2d_gaussian(cx, cy, width, height):
    out = create_2d_gaussian(cx, cy, width, height)
    # Shape should be correct and center point should be set to 1 (largest value)
    assert out.shape == (height, width)
    assert math.isclose(out[cy, cx], 1.0, abs_tol=SMALL, rel_tol=SMALL)
    assert out[cy, cx] == out.flatten().max()


def test_create_2d_gaussian_invalid_inputs():
    with pytest.raises(ValueError, match="X coordinate is outside of width!"):
        _ = create_2d_gaussian(100, 50, 50, 100)
    with pytest.raises(ValueError, match="Y coordinate is outside of height!"):
        _ = create_2d_gaussian(50, 100, 100, 50)


@pytest.mark.parametrize(
    "cx,cy,n_contours",
    [
        # center point in middle of image, should have one contour
        (0.5, 0.5, 1),
        # center point at left edge, should have 2 contours
        (0.05, 0.5, 2),
        # center point at right edge, should have 2
        (0.95, 0.5, 2),
        # center point at top, should have one
        (0.5, 0.95, 1),
        # center point at bottom, should have one
        (0.5, 0.05, 1),
    ],
)
@pytest.mark.parametrize(
    "width,height", [(960, 480), (1920, 960), (400, 200), (480, 240)]
)
def test_get_contours(cx, cy, n_contours, width, height):
    cx_p = int(cx * width)
    cy_p = int(cy * height)

    # dumbass check that the test inputs are correct
    #  i wasted one hour trying to fix this so this is staying
    assert 0 <= cx_p < width
    assert 0 <= cy_p < height

    # create the gaussian with given center and mask values below threshold
    gauss = create_2d_gaussian(cx_p, cy_p, width, height)
    gauss_masked = gauss.copy()
    gauss_masked[np.where(gauss_masked < SMALL)] = 0

    # find the wrapped contours, should have expected number
    conts = find_contours(gauss_masked)
    assert len(conts) == n_contours

    # at most two contours (wrapping left/right)
    assert len(conts) <= 2

    for contour in conts:
        # no degenerate contours
        assert contour.ndim == 2

        # get pixels for the contours
        pxls = np.array(get_segmentation_pixels(gauss_masked, contour))
        assert pxls.shape[-1] == 3

        # check x/y coordinates of contour are in boundary of image
        pxls_x = pxls[:, 0]
        pxls_y = pxls[:, 1]
        assert np.logical_and(0 <= pxls_x, pxls_x <= width).all()
        assert np.logical_and(0 <= pxls_y, pxls_y <= height).all()

    # special case for two contours
    if len(conts) == 2:
        # grab both contour X coordinates
        cnt1, cnt2 = conts
        cnt1_x = cnt1[:, 0]
        cnt2_x = cnt2[:, 0]

        # compute difference between median X coordinate
        #  this should be quite large, reflects that both coordinates are on opposite ends of the screen
        cntd = np.abs(np.median(cnt1_x) - np.median(cnt2_x))
        assert cntd > width // 2


@pytest.mark.parametrize(
    "gauss_x,gauss_y",
    # Position as PROPORTION of actual width/height
    [
        # At top right edge
        (0.95, 0.75),
        # At bottom left edge
        (0.05, 0.25),
        # At top
        (0.5, 0.95),
        # At bottom
        (0.5, 0.05),
    ],
)
@pytest.mark.parametrize("width, height", [(1920, 960), (960, 480), (480, 240)])
def test_equirectangular_contour_area(gauss_x, gauss_y, width, height):
    """
    Due to the equirectangular format, 2D Gaussians at the extreme edges (left/right/upper/lower) should be
    'larger' than those drawn directly in the center.

    We can test this by computing the area of the Gaussian, after masking
    """
    # Compute the Gaussian at the target position
    gauss_cx = int(gauss_x * width)
    gauss_cy = int(gauss_y * height)
    gauss_large = create_2d_gaussian(gauss_cx, gauss_cy, width, height)
    gauss_large_masked = (gauss_large > 4e-5).astype(bool)

    # Sanity check that target position is within mask!
    assert bool(gauss_large_masked[gauss_cy, gauss_cx]) is True

    # Compute the Gaussian at the center of the frame
    gauss_small = create_2d_gaussian(width // 2, height // 2, width, height)
    gauss_small_masked = (gauss_small > 4e-5).astype(bool)

    # The area of the large Gaussian (at the edges of the frame) should be larger
    #  than the area of the small Gaussian (at the center of the frame)
    assert gauss_large_masked.sum() > gauss_small_masked.sum()


# Be warned, here lies LLM-generated tests :`)
@pytest.mark.parametrize(
    "s, a, x",
    [
        # Small 2x2 Hermitian matrix, A 2x2, x as 2x1 vector
        (
            np.array([[1, 2 + 1j], [2 - 1j, 3]]),
            np.array([[1, 0], [0, 1]]),
            np.array([1, 2]),
        ),
        # 3x3 Hermitian, A 3x2, x 2x1
        (
            np.array([[1, 1j, 0], [-1j, 2, 0], [0, 0, 3]]),
            np.array([[1, 0], [0, 1], [1, 1]]),
            np.array([0.5, -0.5]),
        ),
    ],
)
def test_l2_loss(s, a, x):
    # Initialize
    loss = L2Loss(s, a)

    # Test evaluation
    val = loss._eval(x)
    assert isinstance(
        val, (float, np.floating)
    ), "Evaluation should return a scalar float."

    # Test gradient shape
    grad = loss._grad(x)
    assert grad.shape == x.shape, "Gradient shape must match input x."

    # Check gradient is real
    assert np.all(np.isreal(grad)), "Gradient must be real."


def test_l2_loss_invalid_inputs():
    # 1. Shape mismatch: S not square or not matching A
    a = np.ones((2, 3))
    s_wrong_shape = np.ones((3, 2))
    with pytest.raises(ValueError, match="Parameters `s` and `a` are inconsistent."):
        L2Loss(s_wrong_shape, a)

    # 2. Non-Hermitian S
    a = np.ones((2, 2))
    s_non_herm = np.array([[1, 2], [3, 4]])  # Not Hermitian
    with pytest.raises(ValueError, match="Parameter `s` must be Hermitian."):
        L2Loss(s_non_herm, a)


@pytest.mark.parametrize(
    "lambda_, gamma, x, alpha",
    [
        (0.1, 0.5, np.array([1.0, -2.0]), 0.1),
        (1.0, 0.0, np.array([0.5, -0.5, 1.5]), 0.2),
        (0.5, 1.0, np.array([2.0, -1.0]), 0.5),
    ],
)
def test_elastic_net_loss(lambda_, gamma, x, alpha):
    loss = ElasticNetLoss(lambda_, gamma)

    # Test evaluation returns scalar
    val = loss._eval(x)
    assert isinstance(
        val, (float, np.floating)
    ), "Evaluation should return a scalar float."

    # Test proximal operator returns same shape
    prox = loss._prox(x, alpha)
    assert prox.shape == x.shape, "Proximal operator must preserve shape."


def test_elastic_net_loss_invalid_inputs():
    with pytest.raises(ValueError, match="Parameter `lambda_`"):
        _ = ElasticNetLoss(-0.1, None)

    with pytest.raises(ValueError, match="Parameter `gamma`"):
        _ = ElasticNetLoss(0.1, -1)


@pytest.mark.parametrize(
    "d, l_, momentum",
    [
        (2, 1.0, True),
        (5, 0.1, False),
        (3, 10.0, True),
    ],
)
def test_ground_truth_accel(d, l_, momentum):
    accel = GroundTruthAccel(d, l_, momentum)

    class DummySolver:
        def __init__(self, sol):
            self.sol = sol

    sol0 = np.array([1.0, 2.0])
    solver = DummySolver(sol0.copy())

    # Test update_step returns correct step
    step = accel._update_step(solver, None, niter=1)
    assert step == 1 / l_, "Step size must match 1/l_."

    # Test update_sol returns correct shape
    sol_new = accel._update_sol(solver, None, niter=2)
    assert (
        sol_new.shape == sol0.shape
    ), "Updated solution must have same shape as input."

    assert accel._post() == None  # noqa
    assert accel._pre(None, None) == None  # noqa


@pytest.mark.parametrize(
    "s, a, kwargs, match",
    [
        # s not square / inconsistent with a
        (
            np.ones((2, 3)),
            np.ones((2, 3)),
            {},
            "Parameters `s` and `a` are inconsistent.",
        ),
        # s not Hermitian
        (
            np.array([[1, 2], [3, 4]]),
            np.ones((2, 2)),
            {},
            "Parameter `s` must be Hermitian.",
        ),
        # gamma out of range
        (
            np.eye(2),
            np.ones((2, 2)),
            {"gamma": -0.1},
            "Parameter `gamma` is must lie in \\[0, 1\\]",
        ),
        (
            np.eye(2),
            np.ones((2, 2)),
            {"gamma": 1.5},
            "Parameter `gamma` is must lie in \\[0, 1\\]",
        ),
        # l_ <= 0
        (np.eye(2), np.ones((2, 2)), {"l_": 0}, "Parameter `l_` must be positive."),
        # d < 2
        (np.eye(2), np.ones((2, 2)), {"d": 1}, r"Parameter\[d\] must be \\ge 2"),
        # x0 negative
        (
            np.eye(2),
            np.ones((2, 2)),
            {"x0": np.array([-1, 0])},
            "Parameter `x0` must be non-negative.",
        ),
        # eps out of range
        (
            np.eye(2),
            np.ones((2, 2)),
            {"eps": 0},
            "Parameter `eps` must lie in \\(0, 1\\)",
        ),
        (
            np.eye(2),
            np.ones((2, 2)),
            {"eps": 1},
            "Parameter `eps` must lie in \\(0, 1\\)",
        ),
        # n_iter_max < 1
        (
            np.eye(2),
            np.ones((2, 2)),
            {"n_iter_max": 0},
            "Parameter `N_iter_max` must be positive.",
        ),
        # lambda_ negative
        (
            np.eye(2),
            np.ones((2, 2)),
            {"lambda_": -0.1},
            "Parameter `lambda_` must be non-negative.",
        ),
    ],
)
def test_solve_invalid_inputs(s, a, kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve(s, a, **kwargs)


def test_generate_acoustic_image_json_invalid_inputs():
    with pytest.raises(
        ValueError, match="Expected acoustic image to have 3 dimensions"
    ):
        _ = generate_acoustic_image_json(np.zeros((2, 2)), np.zeros((2, 2)))


def test_eigh_max_invalid_inputs():
    with pytest.raises(ValueError, match="`a` has wrong dimensions"):
        _ = eigh_max(np.zeros((2,)))


@pytest.mark.parametrize(
    "acoustic_image",
    [
        # Just some dummy labels I made up
        [
            {
                "metadata_frame_index": 4,
                "instance_id": 0,
                "category_id": 3,
                "segmentation": [],
                "distance": 296.0,
            },
            {
                "metadata_frame_index": 5,
                "instance_id": 0,
                "category_id": 3,
                "segmentation": [
                    [
                        [31, 172, 4.0364242363765154e-05],
                        [32, 172, 4.070701469734546e-05],
                        [33, 172, 4.092166246117612e-05],
                        [34, 172, 4.100662574964808e-05],
                        [35, 172, 4.096155773159229e-05],
                    ],
                    [
                        [22, 183, 5.476282593022971e-05],
                        [23, 183, 5.729792870801646e-05],
                        [24, 183, 5.972850381068057e-05],
                        [25, 183, 6.203015096749988e-05],
                        [26, 183, 6.372703554105687e-05],
                        [27, 183, 6.525805978087924e-05],
                        [28, 183, 6.661017169406419e-05],
                    ],
                ],
                "distance": 296.0,
            },
            {
                "metadata_frame_index": 5,
                "instance_id": 0,
                "category_id": 3,
                "segmentation": [
                    [
                        [46, 204, 4.317460756407866e-05],
                        [47, 204, 4.119375333057327e-05],
                        [23, 205, 4.173396071277285e-05],
                        [24, 205, 4.407868843951187e-05],
                        [25, 205, 4.6344364875221856e-05],
                        [26, 205, 4.8512154691918684e-05],
                        [27, 205, 5.0421919736560936e-05],
                        [28, 205, 5.1311185948953864e-05],
                    ]
                ],
                "distance": 297.0,
            },
        ]
    ],
)
def test_standardise_acoustic_image_amplitude(acoustic_image: list[dict]):
    std = standardise_acoustic_image_amplitude(acoustic_image)

    for lab in std:
        # Should have no more than 2 polygons
        assert len(lab["segmentation"]) <= 2

        for poly in lab["segmentation"]:
            poly_arr = np.array(poly)
            assert poly_arr.shape[-1] == 3

            # Check ranges are correct and scaled
            poly_amp = poly_arr[:, -1]
            assert bool(np.all(np.logical_and(poly_amp <= 1.0, poly_amp >= 0.01)))
