import os
import numpy as np
import matplotlib.pyplot as plt

from decaymc.constants import M_MU, M_E
from decaymc.decay import decay_three_body_rest_via_X, sample_muon_decay_michel
from decaymc.kinematics import spatial, p4, conservation_residual

OUT_DATA = "paper/data"
OUT_FIGS = "paper/figures"

def ensure_dirs():
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)

def analytic_michel_pdf_E(E, m_mu):
    """
    Michel spectrum (massless e approximation):
      (1/Γ) dΓ/dx = x^2(3-2x), x=2E/m_mu in [0,1]
    Convert to density in E: pdf_E = pdf_x * dx/dE = pdf_x * (2/m_mu)
    """
    x = 2.0 * E / m_mu
    pdf_x = x**2 * (3.0 - 2.0 * x)
    pdf_x[(x < 0.0) | (x > 1.0)] = 0.0
    return pdf_x * (2.0 / m_mu)

def main():
    ensure_dirs()
    rng = np.random.default_rng(0)

    N = 200_000

    # Parent at rest
    P_parent = p4(M_MU, np.array([0.0, 0.0, 0.0]))

    # Arrays
    E_phase = np.empty(N)
    E_michel = np.empty(N)

    cos_phase = np.empty(N)
    cos_michel = np.empty(N)

    dE_phase = np.empty(N)
    dP_phase = np.empty(N)

    dE_michel = np.empty(N)
    dP_michel = np.empty(N)

    # --- Phase-space sample ---
    for i in range(N):
        P_e, P_nu1, P_nu2, _ = decay_three_body_rest_via_X(M_MU, M_E, 0.0, 0.0, rng=rng)
        E_phase[i] = P_e[0]

        pe = spatial(P_e)
        cos_phase[i] = pe[2] / np.linalg.norm(pe)

        res = conservation_residual(P_parent, [P_e, P_nu1, P_nu2])
        dE_phase[i] = res[0]
        dP_phase[i] = np.linalg.norm(res[1:])

    # --- Michel-weighted sample ---
    for i in range(N):
        P_e, P_nu1, P_nu2, _ = sample_muon_decay_michel(M_MU, M_E, rng=rng)
        E_michel[i] = P_e[0]

        pe = spatial(P_e)
        cos_michel[i] = pe[2] / np.linalg.norm(pe)

        res = conservation_residual(P_parent, [P_e, P_nu1, P_nu2])
        dE_michel[i] = res[0]
        dP_michel[i] = np.linalg.norm(res[1:])

    # Save raw data for reproducibility
    np.savez(
        os.path.join(OUT_DATA, "muon_spectra.npz"),
        N=N,
        M_MU=M_MU,
        M_E=M_E,
        E_phase=E_phase,
        E_michel=E_michel,
        cos_phase=cos_phase,
        cos_michel=cos_michel,
        dE_phase=dE_phase,
        dP_phase=dP_phase,
        dE_michel=dE_michel,
        dP_michel=dP_michel,
    )

    # ----------------------------
    # Figure D1: Michel spectrum overlay
    # ----------------------------
    E_min = M_E
    E_max = M_MU / 2.0  # Michel massless endpoint

    # Clip to Michel range for fair overlay with analytic curve
    E_phase_clip = E_phase[(E_phase >= E_min) & (E_phase <= E_max)]
    E_michel_clip = E_michel[(E_michel >= E_min) & (E_michel <= E_max)]

    bins = 140

    plt.figure()
    plt.hist(E_phase_clip, bins=bins, range=(E_min, E_max), density=True, alpha=0.55, label="Phase-space")
    plt.hist(E_michel_clip, bins=bins, range=(E_min, E_max), density=True, alpha=0.55, label="Michel-weighted")

    E_grid = np.linspace(E_min, E_max, 800)
    pdf_E = analytic_michel_pdf_E(E_grid, M_MU)
    plt.plot(E_grid, pdf_E, label="Analytic Michel (massless e)")

    plt.xlabel(r"Electron energy $E_e$ (MeV)")
    plt.ylabel("Probability density")
    plt.title("Muon decay: phase-space vs Michel-weighted electron spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_michel_spectrum.pdf"))
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_michel_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Figure D2: cos(theta) isotropy check
    # ----------------------------
    plt.figure()
    plt.hist(cos_phase, bins=80, range=(-1, 1), density=True, alpha=0.55, label=r"Phase-space $\cos\theta_e$")
    plt.hist(cos_michel, bins=80, range=(-1, 1), density=True, alpha=0.55, label=r"Michel-weighted $\cos\theta_e$")
    plt.xlabel(r"$\cos\theta_e$")
    plt.ylabel("Probability density")
    plt.title("Muon decay: electron direction isotropy check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_cos_theta.pdf"))
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_cos_theta.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Figure D3: conservation diagnostics
    # ----------------------------
    plt.figure()
    plt.hist(np.abs(dE_phase), bins=120, density=True, alpha=0.55, label=r"Phase-space $|\Delta E|$")
    plt.hist(np.abs(dE_michel), bins=120, density=True, alpha=0.55, label=r"Michel $|\Delta E|$")
    plt.xlabel(r"$|\Delta E|$ (MeV)")
    plt.ylabel("Probability density")
    plt.title("Muon decay: energy conservation residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_conservation_deltaE.pdf"))
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_conservation_deltaE.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(dP_phase, bins=120, density=True, alpha=0.55, label=r"Phase-space $|\Delta \vec{p}|$")
    plt.hist(dP_michel, bins=120, density=True, alpha=0.55, label=r"Michel $|\Delta \vec{p}|$")
    plt.xlabel(r"$|\Delta \vec{p}|$ (MeV)")
    plt.ylabel("Probability density")
    plt.title("Muon decay: momentum conservation residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_conservation_deltaP.pdf"))
    plt.savefig(os.path.join(OUT_FIGS, "fig_muon_conservation_deltaP.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Wrote:")
    print("  paper/data/muon_spectra.npz")
    print("  paper/figures/fig_muon_michel_spectrum.pdf")
    print("  paper/figures/fig_muon_michel_spectrum.png")
    print("  paper/figures/fig_muon_cos_theta.pdf")
    print("  paper/figures/fig_muon_cos_theta.png")
    print("  paper/figures/fig_muon_conservation_deltaE.pdf")
    print("  paper/figures/fig_muon_conservation_deltaE.png")
    print("  paper/figures/fig_muon_conservation_deltaP.pdf")
    print("  paper/figures/fig_muon_conservation_deltaP.png")

if __name__ == "__main__":
    main()

