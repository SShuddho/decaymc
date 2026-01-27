# decaymc

[![CI](https://github.com/SShuddho/decaymc/actions/workflows/ci.yml/badge.svg)](https://github.com/SShuddho/decaymc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)



A simple Monte Carlo framework for relativistic particle decays and scattering,
designed to emphasize physical transparency, explicit assumptions, and validation against
analytic benchmarks.

---

## Motivation

Modern particle-physics simulations are typically performed using large, highly optimized
event generators that act as black boxes. While indispensable for experimental analyses,
such tools can obscure the connection between fundamental physics, numerical methods, and
the generated events.

This project aims to provide a minimal, transparent alternative that demonstrates how
relativistic particle decays and scattering processes can be simulated directly from
first principles using Monte Carlo techniques, without relying on specialized HEP
frameworks.

---

## Scope

All quantities are expressed in natural units with energies and masses in MeV.
Unless otherwise stated, decays are generated in the parent rest frame.

### Relativistic Kinematics
- Explicit four-vector representation with metric signature (+, −, −, −)
- Lorentz boosts implemented from first principles
- Invariant mass calculations and on-shell checks
- Event-level energy–momentum conservation verification

### Particle Decays
- General two-body decays with isotropic emission
- Three-body decays via an intermediate system (phase-space–style proposal)
- Muon decay:
  - phase-space generation
  - Michel-spectrum weighting for the electron energy distribution
- Validation against analytic kinematic bounds and expected distributions

### Scattering
- Rutherford scattering in scaled (arbitrary) units
- Angular acceptance cuts to regularize the forward divergence
- Total cross-sections computed via:
  - numerical quadrature (benchmark)
  - Monte Carlo integration with uncertainty estimates
- Rejection-free importance sampling for Rutherford angular distributions

---

## Non-Goals and Limitations

This framework intentionally does **not** include:
- spin or polarization correlations (except for Michel energy weighting),
- radiative corrections,
- detector effects or resolution modeling,
- hadronic or nuclear form factors,
- high-performance or large-scale event generation.

The focus is methodological clarity rather than phenomenological completeness.

---

## Numerical and Design Philosophy

The design philosophy:
- physical assumptions are stated explicitly,
- numerical methods are implemented transparently,
- validation precedes optimization,
- dependencies are kept minimal (NumPy and SciPy only).

Where naive Monte Carlo sampling is inefficient, importance sampling strategies are derived
and implemented explicitly.


---

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

## Quickstart

Generate a Michel-weighted muon decay event in the muon rest frame and verify
four-momentum conservation.

```python
import numpy as np

from decaymc.decay import sample_muon_decay_michel, check_conservation_for_decay
from decaymc.constants import M_MU, M_E

rng = np.random.default_rng(123)

P_e, P_nu1, P_nu2, tries = sample_muon_decay_michel(M_MU, M_E, rng=rng)

res = check_conservation_for_decay(M_MU, [P_e, P_nu1, P_nu2])

print("tries:", tries)
print("electron four-momentum [E, px, py, pz] (MeV):", P_e)
print("conservation residual [dE, dpx, dpy, dpz]:", res)
print("max |residual|:", np.max(np.abs(res)))
```


---

## License

This project is released under the MIT License. See the LICENSE file for details.

---

## Citing

If you use this software, please cite it using the provided `CITATION.cff` file.
