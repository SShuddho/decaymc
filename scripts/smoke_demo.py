import numpy as np
from decaymc.constants import M_MU, M_E
from decaymc.decay import sample_muon_decay_michel
from decaymc.scattering import sigma_quadrature_rutherford

rng = np.random.default_rng(0)
P_e, P_nu1, P_nu2, tries = sample_muon_decay_michel(M_MU, M_E, rng=rng)
print("Muon Michel demo: E_e =", P_e[0], "MeV | tries =", tries)

theta_min = np.deg2rad(5.0)
theta_max = np.deg2rad(175.0)
sigma, err = sigma_quadrature_rutherford(theta_min, theta_max, A=1.0)
print("Rutherford sigma (scaled):", sigma, "+/-", err)

