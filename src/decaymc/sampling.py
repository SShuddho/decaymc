"""
Sampling utilities (physics-agnostic).

Includes:
- isotropic direction sampling
- uniform sampling in solid angle within theta cuts
- generic accept-reject sampler
- Rutherford-specific importance sampling for theta
"""

from __future__ import annotations
import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def sample_isotropic_unit_vector(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample an isotropic unit vector uniformly on the sphere.
    """
    if rng is None:
        rng = np.random.default_rng()

    phi = 2.0 * np.pi * rng.random()
    cos_theta = 2.0 * rng.random() - 1.0
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], dtype=float)


def sample_uniform_solid_angle(
    theta_min: float,
    theta_max: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample (theta, phi) uniformly in solid angle within theta ∈ [theta_min, theta_max].

    Uniform in solid angle:
      phi ~ U[0, 2π)
      cos(theta) ~ U[cos(theta_max), cos(theta_min)]
    """
    if rng is None:
        rng = np.random.default_rng()

    phi = 2.0 * np.pi * rng.random(n)
    u = rng.random(n)

    cos_min = np.cos(theta_max)
    cos_max = np.cos(theta_min)
    cos_theta = cos_min + (cos_max - cos_min) * u
    theta = np.arccos(cos_theta)
    return theta, phi


def accept_reject_1d(
    pdf,
    x_min: float,
    x_max: float,
    pdf_max: float,
    rng: np.random.Generator | None = None,
    max_tries: int = 1_000_000,
) -> tuple[float, int]:
    """
    Generic 1D accept-reject sampler for unnormalized pdf(x).
    """
    if rng is None:
        rng = np.random.default_rng()

    for tries in range(max_tries):
        x = x_min + (x_max - x_min) * rng.random()
        y = pdf_max * rng.random()
        if y <= pdf(x):
            return x, tries
    raise RuntimeError("accept_reject_1d failed: check pdf_max or pdf behavior.")


def sample_rutherford_theta_importance(
    n: int,
    theta_min: float,
    theta_max: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Rejection-free inverse-CDF sampling for Rutherford theta marginal:

      p(theta) ∝ csc^4(theta/2) * sin(theta),  theta ∈ [theta_min, theta_max].

    Derivation yields:
      Let u = sin(theta/2), then 1/u^2 is linear in a uniform variate.
    """
    if rng is None:
        rng = np.random.default_rng()

    umin = np.sin(theta_min / 2.0)
    umax = np.sin(theta_max / 2.0)

    if not (umin > 0.0 and umax > 0.0 and umin < umax):
        raise ValueError("Invalid theta bounds for Rutherford importance sampling.")

    r = rng.random(n)
    inv_u2 = (1.0 - r) / (umin * umin) + r / (umax * umax)
    u = 1.0 / np.sqrt(inv_u2)
    theta = 2.0 * np.arcsin(u)
    return theta

