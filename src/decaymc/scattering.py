"""
Scattering utilities: Rutherford scattering in scaled units with angular cuts.

Includes:
- dσ/dΩ (scaled)
- quadrature benchmark for σ
- Monte Carlo integration for σ
- rejection-free Rutherford angle generator (importance sampling)
"""

from __future__ import annotations
import numpy as np
from scipy import integrate

from .sampling import sample_uniform_solid_angle, sample_rutherford_theta_importance


def dsigma_dOmega_rutherford(theta: np.ndarray, A: float = 1.0) -> np.ndarray:
    """
    Rutherford angular dependence (scaled units):
      dσ/dΩ = A * csc^4(theta/2)
    """
    s = np.sin(theta / 2.0)
    return A / (s**4)


def sigma_quadrature_rutherford(theta_min: float, theta_max: float, A: float = 1.0) -> tuple[float, float]:
    """
    σ = ∫_{θmin}^{θmax} 2π (dσ/dΩ)(θ) sinθ dθ
    Returns (sigma, quad_abs_error_est).
    """
    def integrand(theta):
        return 2.0 * np.pi * dsigma_dOmega_rutherford(theta, A=A) * np.sin(theta)

    sigma, err = integrate.quad(integrand, theta_min, theta_max, limit=300)
    return float(sigma), float(err)


def sigma_mc_rutherford(theta_min: float, theta_max: float, N: int, A: float = 1.0, rng: np.random.Generator | None = None) -> tuple[float, float]:
    """
    Monte Carlo estimate of σ using uniform-in-solid-angle sampling over acceptance:
      σ ≈ Ω_acc * mean[dσ/dΩ(theta_i)]
    Also returns a standard-error estimate from sample variance.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta, _ = sample_uniform_solid_angle(theta_min, theta_max, N, rng=rng)
    w = dsigma_dOmega_rutherford(theta, A=A)

    Omega_acc = 2.0 * np.pi * (np.cos(theta_min) - np.cos(theta_max))
    sigma_hat = Omega_acc * float(np.mean(w))

    sigma_w = float(np.std(w, ddof=1))
    sigma_err = Omega_acc * sigma_w / np.sqrt(N)
    return sigma_hat, float(sigma_err)


def sample_rutherford_angles(theta_min: float, theta_max: float, N: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Rejection-free Rutherford angle generator over [theta_min, theta_max]:
      theta ~ p(theta) ∝ (dσ/dΩ)(theta) sinθ
      phi ~ Uniform(0, 2π)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = sample_rutherford_theta_importance(N, theta_min, theta_max, rng=rng)
    phi = 2.0 * np.pi * rng.random(N)
    return theta, phi

