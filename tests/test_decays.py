import numpy as np
from decaymc.constants import M_MU, M_E, M_PI0, M_PIP
from decaymc.decay import (
    decay_two_body_rest, decay_three_body_rest_via_X,
    sample_muon_decay_michel, check_conservation_for_decay
)

def test_pi0_to_gg_energies():
    rng = np.random.default_rng(0)
    P1, P2 = decay_two_body_rest(M_PI0, 0.0, 0.0, rng=rng)
    assert np.isclose(P1[0], M_PI0/2.0, atol=1e-12)
    assert np.isclose(P2[0], M_PI0/2.0, atol=1e-12)

def test_pip_to_mu_nu_energies():
    rng = np.random.default_rng(1)
    Pmu, Pnu = decay_two_body_rest(M_PIP, M_MU, 0.0, rng=rng)
    E_mu = (M_PIP**2 + M_MU**2) / (2.0 * M_PIP)
    E_nu = (M_PIP**2 - M_MU**2) / (2.0 * M_PIP)
    assert np.isclose(Pmu[0], E_mu, atol=1e-12)
    assert np.isclose(Pnu[0], E_nu, atol=1e-12)

def test_three_body_conservation():
    rng = np.random.default_rng(2)
    P1, P2, P3, _ = decay_three_body_rest_via_X(M_MU, M_E, 0.0, 0.0, rng=rng)
    res = check_conservation_for_decay(M_MU, [P1, P2, P3])
    assert np.max(np.abs(res)) < 1e-9

def test_michel_generator_x_in_range_and_conservation():
    rng = np.random.default_rng(3)
    P_e, P_nu1, P_nu2, _ = sample_muon_decay_michel(M_MU, M_E, rng=rng)
    x = 2.0 * P_e[0] / M_MU
    assert 0.0 <= x <= 1.0
    res = check_conservation_for_decay(M_MU, [P_e, P_nu1, P_nu2])
    assert np.max(np.abs(res)) < 1e-9

