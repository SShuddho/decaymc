"""
Relativistic kinematics primitives (physics-agnostic).

Conventions:
- 4-vector stored as numpy array [E, px, py, pz]
- Minkowski metric (+, -, -, -)
"""

from __future__ import annotations
import numpy as np


def p4(E: float, p_vec: np.ndarray) -> np.ndarray:
    return np.array([E, float(p_vec[0]), float(p_vec[1]), float(p_vec[2])], dtype=float)


def spatial(p4_vec: np.ndarray) -> np.ndarray:
    return p4_vec[1:]


def minkowski_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] - np.dot(a[1:], b[1:]))


def invariant_mass_squared(p4_vec: np.ndarray) -> float:
    return minkowski_dot(p4_vec, p4_vec)


def invariant_mass(p4_vec: np.ndarray) -> float:
    return float(np.sqrt(max(invariant_mass_squared(p4_vec), 0.0)))


def on_shell_momentum_magnitude(E: float, m: float) -> float:
    return float(np.sqrt(max(E * E - m * m, 0.0)))


def lorentz_boost(p4_vec: np.ndarray, beta_vec: np.ndarray) -> np.ndarray:
    """
    Boost 4-vector by 3-velocity beta_vec. Requires |beta| < 1.
    """
    beta = np.asarray(beta_vec, dtype=float)
    b2 = float(np.dot(beta, beta))

    if b2 == 0.0:
        return p4_vec.copy()
    if b2 >= 1.0:
        raise ValueError("Invalid boost: |beta| must be < 1.")

    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = float(np.dot(beta, p4_vec[1:]))

    E_prime = gamma * (p4_vec[0] + bp)
    factor = ((gamma - 1.0) * bp / b2) + gamma * p4_vec[0]
    p_prime = p4_vec[1:] + factor * beta

    return np.array([E_prime, p_prime[0], p_prime[1], p_prime[2]], dtype=float)


def total_four_momentum(p4_list: list[np.ndarray]) -> np.ndarray:
    return np.sum(np.array(p4_list, dtype=float), axis=0)


def conservation_residual(p_initial: np.ndarray, p_final_list: list[np.ndarray]) -> np.ndarray:
    return total_four_momentum(p_final_list) - p_initial

