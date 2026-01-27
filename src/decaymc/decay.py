"""
Decay generators (parent at rest).

Includes:
- two-body decay in rest frame
- three-body decay via intermediate system X (phase-space-style proposal)
- Michel-weighted muon decay generator (energy-spectrum validation)
"""

from __future__ import annotations
import numpy as np

from .kinematics import (
    p4, spatial, invariant_mass_squared, on_shell_momentum_magnitude,
    lorentz_boost, conservation_residual
)
from .sampling import sample_isotropic_unit_vector


def decay_two_body_rest(
    m_parent: float,
    m1: float,
    m2: float,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parent -> 1 + 2, in parent rest frame.
    """
    if rng is None:
        rng = np.random.default_rng()

    if m_parent < m1 + m2:
        raise ValueError("Kinematically forbidden: m_parent < m1 + m2.")

    m = m_parent
    term1 = m * m - (m1 + m2) ** 2
    term2 = m * m - (m1 - m2) ** 2
    p_mag = float(np.sqrt(max(term1 * term2, 0.0)) / (2.0 * m))

    E1 = float(np.sqrt(p_mag * p_mag + m1 * m1))
    E2 = float(np.sqrt(p_mag * p_mag + m2 * m2))

    n_hat = sample_isotropic_unit_vector(rng)
    p1_vec = p_mag * n_hat
    p2_vec = -p1_vec

    return p4(E1, p1_vec), p4(E2, p2_vec)


def energy_bounds_three_body_rest(m_parent: float, m1: float, m2: float, m3: float) -> tuple[float, float]:
    """
    Bounds for particle-1 energy in Parent -> 1 + 2 + 3, using Parent -> 1 + X picture.
    """
    if m_parent < m1 + m2 + m3:
        raise ValueError("Kinematically forbidden: m_parent < m1 + m2 + m3.")

    E1_min = m1
    E1_max = (m_parent**2 + m1**2 - (m2 + m3)**2) / (2.0 * m_parent)
    return float(E1_min), float(E1_max)


def decay_three_body_rest_via_X(
    m_parent: float,
    m1: float,
    m2: float,
    m3: float,
    rng: np.random.Generator | None = None,
    max_tries: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Phase-space-style 3-body decay in parent rest frame via intermediate X:
      Parent -> 1 + X
      X -> 2 + 3
    Proposal: sample E1 uniformly in allowed range, direction isotropic.
    Accept if mX >= m2+m3.
    """
    if rng is None:
        rng = np.random.default_rng()

    P_parent = p4(m_parent, np.array([0.0, 0.0, 0.0]))
    E1_min, E1_max = energy_bounds_three_body_rest(m_parent, m1, m2, m3)

    for tries in range(max_tries):
        E1 = E1_min + (E1_max - E1_min) * rng.random()
        p1_mag = on_shell_momentum_magnitude(E1, m1)
        p1_vec = p1_mag * sample_isotropic_unit_vector(rng)
        P1 = p4(E1, p1_vec)

        P_X = P_parent - P1
        mX2 = invariant_mass_squared(P_X)

        if mX2 < (m2 + m3) ** 2:
            continue

        mX = float(np.sqrt(max(mX2, 0.0)))
        P2_star, P3_star = decay_two_body_rest(mX, m2, m3, rng=rng)

        beta_vec = spatial(P_X) / P_X[0]
        P2 = lorentz_boost(P2_star, beta_vec)
        P3 = lorentz_boost(P3_star, beta_vec)

        return P1, P2, P3, tries

    raise RuntimeError("decay_three_body_rest_via_X failed: max_tries exceeded.")


def michel_weight_unpolarized(x: float) -> float:
    """
    Michel spectrum (massless e approximation):
      (1/Γ) dΓ/dx = x^2 (3 - 2x),  x ∈ [0,1]
    Used as acceptance probability.
    """
    if x < 0.0 or x > 1.0:
        return 0.0
    return float(x * x * (3.0 - 2.0 * x))


def sample_muon_decay_michel(
    m_mu: float,
    m_e: float,
    rng: np.random.Generator | None = None,
    max_tries: int = 1_000_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Michel-weighted muon decay:
      mu -> e + nu + nu  (neutrinos massless)
    Uses 3-body phase-space generator as proposal and accept-reject on x=2E_e/m_mu.
    """
    if rng is None:
        rng = np.random.default_rng()

    for tries in range(max_tries):
        P_e, P_nu1, P_nu2, _ = decay_three_body_rest_via_X(m_mu, m_e, 0.0, 0.0, rng=rng)
        x = 2.0 * P_e[0] / m_mu
        if x > 1.0:
            continue
        w = michel_weight_unpolarized(x)
        if rng.random() < w:
            return P_e, P_nu1, P_nu2, tries

    raise RuntimeError("sample_muon_decay_michel failed: max_tries exceeded.")


def check_conservation_for_decay(m_parent: float, final: list[np.ndarray]) -> np.ndarray:
    """
    Convenience helper: residual for parent-at-rest decay.
    """
    P_parent = p4(m_parent, np.array([0.0, 0.0, 0.0]))
    return conservation_residual(P_parent, final)

