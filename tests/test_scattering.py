import numpy as np
from decaymc.scattering import sigma_quadrature_rutherford, sigma_mc_rutherford, sample_rutherford_angles

def test_rutherford_sigma_mc_close_to_quad():
    theta_min = np.deg2rad(5.0)
    theta_max = np.deg2rad(175.0)

    sigma_q, _ = sigma_quadrature_rutherford(theta_min, theta_max, A=1.0)
    rng = np.random.default_rng(0)
    sigma_mc, sigma_err = sigma_mc_rutherford(theta_min, theta_max, N=200_000, A=1.0, rng=rng)

    # Allow a few-sigma tolerance due to heavy tail variance
    assert abs(sigma_mc - sigma_q) < 5.0 * sigma_err

def test_rutherford_angle_bounds():
    theta_min = np.deg2rad(5.0)
    theta_max = np.deg2rad(175.0)

    rng = np.random.default_rng(1)
    theta, phi = sample_rutherford_angles(theta_min, theta_max, N=50_000, rng=rng)

    assert theta.min() >= theta_min - 1e-12
    assert theta.max() <= theta_max + 1e-12
    assert phi.min() >= 0.0
    assert phi.max() <= 2.0 * np.pi + 1e-12

