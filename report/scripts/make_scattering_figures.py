import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from particle_mc.scattering import (
    dsigma_dOmega_rutherford,
    sigma_quadrature_rutherford,
    sigma_mc_rutherford,
    sample_rutherford_angles,
)

OUT_DATA = "paper/data"
OUT_FIGS = "paper/figures"

def ensure_dirs():
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)

def main():
    ensure_dirs()
    rng = np.random.default_rng(0)

    # Angular cuts (match your notebook choice)
    theta_min = np.deg2rad(5.0)
    theta_max = np.deg2rad(175.0)
    A = 1.0  # scaled units

    # --------------------------------------------
    # Benchmark sigma by quadrature
    # --------------------------------------------
    sigma_quad, sigma_quad_err = sigma_quadrature_rutherford(theta_min, theta_max, A=A)

    # --------------------------------------------
    # Figure S1: dσ/dΩ(θ) over acceptance
    # --------------------------------------------
    theta_grid = np.linspace(theta_min, theta_max, 1200)
    dsdomega = dsigma_dOmega_rutherford(theta_grid, A=A)

    plt.figure()
    plt.plot(theta_grid, dsdomega)
    plt.yscale("log")
    plt.xlabel(r"Scattering angle $\theta$ (rad)")
    plt.ylabel(r"Scaled $d\sigma/d\Omega$ (arb. units)")
    plt.title(r"Rutherford angular dependence (with acceptance cuts)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_rutherford_dsdomega.pdf"))
    plt.close()

    # --------------------------------------------
    # Figure S2: sigma_MC convergence vs N
    # --------------------------------------------
    N_list = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
    rows = [["N", "sigma_mc", "sigma_mc_err", "sigma_quad", "sigma_quad_err", "rel_error"]]

    sigma_mc_list = []
    sigma_err_list = []

    for N in N_list:
        sigma_mc, sigma_mc_err = sigma_mc_rutherford(theta_min, theta_max, N=N, A=A, rng=rng)
        rel_err = (sigma_mc - sigma_quad) / sigma_quad

        rows.append([N, sigma_mc, sigma_mc_err, sigma_quad, sigma_quad_err, rel_err])
        sigma_mc_list.append(sigma_mc)
        sigma_err_list.append(sigma_mc_err)

    out_csv = os.path.join(OUT_DATA, "rutherford_sigma_convergence.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    plt.figure()
    plt.errorbar(N_list, sigma_mc_list, yerr=sigma_err_list, fmt="o", capsize=3, label=r"$\sigma_{\mathrm{MC}} \pm 1\sigma$")
    plt.axhline(sigma_quad, linestyle="--", label=r"$\sigma_{\mathrm{quad}}$")

    plt.xscale("log")
    plt.xlabel("Number of samples N")
    plt.ylabel(r"Total cross-section $\sigma$ (scaled units)")
    plt.title("Monte Carlo convergence of Rutherford total cross-section")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_rutherford_sigma_convergence.pdf"))
    plt.close()

    # --------------------------------------------
    # Figure S3: theta sampling validation
    # --------------------------------------------
    N_angles = 200_000
    theta_samp, phi_samp = sample_rutherford_angles(theta_min, theta_max, N=N_angles, rng=rng)

    # Theory: p(theta) ∝ (dσ/dΩ)(θ) sinθ, normalized over acceptance
    p_unnorm = dsigma_dOmega_rutherford(theta_grid, A=A) * np.sin(theta_grid)
    norm = np.trapezoid(p_unnorm, theta_grid)
    p_theta = p_unnorm / norm

    np.savez(
        os.path.join(OUT_DATA, "rutherford_theta_samples.npz"),
        theta_min=theta_min,
        theta_max=theta_max,
        A=A,
        sigma_quad=sigma_quad,
        sigma_quad_err=sigma_quad_err,
        theta_grid=theta_grid,
        p_theta=p_theta,
        theta_samp=theta_samp,
        phi_samp=phi_samp,
    )

    plt.figure()
    plt.hist(theta_samp, bins=140, range=(theta_min, theta_max), density=True, alpha=0.6, label="MC samples")
    plt.plot(theta_grid, p_theta, label=r"Theory $\propto (d\sigma/d\Omega)\sin\theta$")
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel("Probability density")
    plt.title(r"Rutherford $\theta$ sampling validation (importance sampling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "fig_rutherford_theta_pdf.pdf"))
    plt.close()

    print("Wrote:")
    print(" ", out_csv)
    print("  paper/data/rutherford_theta_samples.npz")
    print("  paper/figures/fig_rutherford_dsdomega.pdf")
    print("  paper/figures/fig_rutherford_sigma_convergence.pdf")
    print("  paper/figures/fig_rutherford_theta_pdf.pdf")

if __name__ == "__main__":
    main()

