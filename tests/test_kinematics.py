import numpy as np
from decaymc.kinematics import p4, invariant_mass_squared, lorentz_boost

def test_invariant_mass_under_boost():
    P = p4(10.0, np.array([3.0, 4.0, 0.0]))
    m2_before = invariant_mass_squared(P)

    beta = np.array([0.2, 0.1, 0.0])
    Pp = lorentz_boost(P, beta)
    m2_after = invariant_mass_squared(Pp)

    assert np.isclose(m2_before, m2_after, rtol=0, atol=1e-12)

