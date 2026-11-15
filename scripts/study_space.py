"""
Convergence study in space (manufactured case "space", exact in time).

Expected (Lagrange P_k, smooth solution):
    ||u - u_h||_{L2} = O(h^{k+1}),   |u - u_h|_{H1} = O(h^k)
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from common import FIGS, RESULTS, rates, run_manufactured

T, DT, THETA = 0.5, 0.05, 1.0
NS = [4, 8, 16, 32, 64]
DEGREES = [1, 2, 3]

rows = []
for k in DEGREES:
    for N in NS:
        eL2, eH1, ndofs = run_manufactured("space", N, k, DT, T, THETA)
        rows.append(dict(degree=k, N=N, h=1.0 / N, ndofs=ndofs, err_L2=eL2, err_H1=eH1))
        if MPI.COMM_WORLD.rank == 0:
            print(f"P{k}  N={N:3d}  ndofs={ndofs:6d}  L2={eL2:.3e}  H1={eH1:.3e}", flush=True)

if MPI.COMM_WORLD.rank == 0:
    for k in DEGREES:
        sub = [r for r in rows if r["degree"] == k]
        hs = [r["h"] for r in sub]
        for r, rl2, rh1 in zip(sub, rates([r["err_L2"] for r in sub], hs),
                               rates([r["err_H1"] for r in sub], hs)):
            r["rate_L2"], r["rate_H1"] = rl2, rh1

    with open(RESULTS / "convergence_space.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    for ax, key, shift, title in [(axes[0], "err_L2", 1, r"Erreur $L^2$"),
                                  (axes[1], "err_H1", 0, r"Erreur semi-norme $H^1$")]:
        for k in DEGREES:
            sub = [r for r in rows if r["degree"] == k]
            h = np.array([r["h"] for r in sub])
            e = np.array([r[key] for r in sub])
            ax.loglog(h, e, "o-", color=colors[k], label=rf"$\mathbb{{P}}_{k}$")
            p = k + shift
            ref = e[-1] * (h / h[-1]) ** p
            ax.loglog(h, ref * 0.5, "--", color=colors[k], lw=0.9, alpha=0.7)
            ax.annotate(rf"$h^{p}$", (h[1], ref[1] * 0.5), color=colors[k],
                        textcoords="offset points", xytext=(4, -12))
        ax.set_xlabel(r"$h$")
        ax.set_xticks([1 / n for n in NS], [rf"$1/{n}$" for n in NS])
        ax.minorticks_off()
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(rf"Convergence en espace — Euler implicite, $\Delta t={DT}$, $T={T}$")
    fig.tight_layout()
    fig.savefig(FIGS / "convergence_space.png", dpi=160)
    print("-> results/convergence_space.csv, figures/convergence_space.png")
