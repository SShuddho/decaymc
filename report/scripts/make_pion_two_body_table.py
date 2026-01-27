import os
import csv
import numpy as np

from particle_mc.constants import M_MU, M_PI0, M_PIP
from particle_mc.decay import decay_two_body_rest

OUT_DATA = "paper/data"

def ensure_dirs():
    os.makedirs(OUT_DATA, exist_ok=True)

def main():
    ensure_dirs()
    rng = np.random.default_rng(0)

    # One event is enough because energies are fixed in parent rest frame
    # But we still compute from kinematics to show agreement.
    P_g1, P_g2 = decay_two_body_rest(M_PI0, 0.0, 0.0, rng=rng)
    E_gamma_sim = P_g1[0]
    E_gamma_ana = M_PI0 / 2.0

    P_mu, P_nu = decay_two_body_rest(M_PIP, M_MU, 0.0, rng=rng)
    E_mu_sim = P_mu[0]
    E_nu_sim = P_nu[0]

    E_mu_ana = (M_PIP**2 + M_MU**2) / (2.0 * M_PIP)
    E_nu_ana = (M_PIP**2 - M_MU**2) / (2.0 * M_PIP)

    rows = [
        ["Decay", "Quantity", "Analytic (MeV)", "Simulation (MeV)", "Abs diff (MeV)"],
        [r"pi0 -> gamma gamma", r"E_gamma", f"{E_gamma_ana:.12f}", f"{E_gamma_sim:.12f}", f"{abs(E_gamma_sim-E_gamma_ana):.3e}"],
        [r"pi+ -> mu+ nu", r"E_mu", f"{E_mu_ana:.12f}", f"{E_mu_sim:.12f}", f"{abs(E_mu_sim-E_mu_ana):.3e}"],
        [r"pi+ -> mu+ nu", r"E_nu", f"{E_nu_ana:.12f}", f"{E_nu_sim:.12f}", f"{abs(E_nu_sim-E_nu_ana):.3e}"],
    ]

    out_csv = os.path.join(OUT_DATA, "pion_two_body_table.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()

