"""
Convergence study in time (manufactured case "time").

Fine P3 mesh (N = 48) so that the space error (~1e-8) is very small.
Expected:  backward Euler (theta=1) -> O(dt),  Crank-Nicolson (theta=1/2) -> O(dt^2)
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from common import FIGS, RESULTS, rates, run_manufactured

T, N, DEG = 1.0, 48, 3
DTS = [0.2 / 2**i for i in range(6)]
SCHEMES = {1.0: "Euler implicite", 0.5: "Crank-Nicolson"}

rows = []
for theta, name in SCHEMES.items():
    for dt in DTS:
        eL2, eH1, _ = run_manufactured("time", N, DEG, dt, T, theta)
        rows.append(dict(theta=theta, scheme=name, dt=dt, n_steps=int(round(T / dt)),
                         err_L2=eL2, err_H1=eH1))
        if MPI.COMM_WORLD.rank == 0:
            print(f"{name:16s} dt={dt:.5f}  L2={eL2:.3e}", flush=True)

if MPI.COMM_WORLD.rank == 0:
    for theta in SCHEMES:
        sub = [r for r in rows if r["theta"] == theta]
        for r, rr in zip(sub, rates([r["err_L2"] for r in sub], [r["dt"] for r in sub])):
            r["rate_L2"] = rr

    with open(RESULTS / "convergence_time.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for (theta, name), c, p in zip(SCHEMES.items(), ["#1f77b4", "#d62728"], [1, 2]):
        sub = [r for r in rows if r["theta"] == theta]
        dt = np.array([r["dt"] for r in sub])
        e = np.array([r["err_L2"] for r in sub])
        ax.loglog(dt, e, "o-", color=c, label=rf"{name} ($\theta={theta:g}$)")
        ref = e[0] * (dt / dt[0]) ** p * 1.6
        ax.loglog(dt, ref, "--", color=c, lw=0.9, alpha=0.7)
        ax.annotate(rf"$\Delta t^{p}$", (dt[2], ref[2]), color=c,
                    textcoords="offset points", xytext=(4, 6))
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\|u(T)-u_h^N\|_{L^2}$")
    ax.set_title(rf"Convergence en temps — $\mathbb{{P}}_{DEG}$, $N={N}$, $T={T:g}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "convergence_time.png", dpi=160)
    print("-> results/convergence_time.csv, figures/convergence_time.png")
