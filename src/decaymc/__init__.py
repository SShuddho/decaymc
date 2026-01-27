"""
decaymc: Simple Monte Carlo utilities for relativistic decays and scattering.

Core modules:
- sampling: random sampling utilities
- kinematics: four-vectors and Lorentz boosts
- decay: 2-body / 3-body decays + Michel-weighted muon decay
- scattering: Rutherford scattering with cuts + MC integration
"""

from . import constants, sampling, kinematics, decay, scattering, utils

__all__ = ["constants", "sampling", "kinematics", "decay", "scattering", "utils"]

