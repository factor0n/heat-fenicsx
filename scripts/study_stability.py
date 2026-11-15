"""
Stability study of the theta scheme.

A) Forward Euler (theta = 0) is stable only if dt <= dt_c = 2 / (kappa * lambda_max),
   where lambda_max is the largest eigenvalue of the problem K x = lambda M x.
   We compute lambda_max, then run with dt = 0.95 dt_c and dt = 1.05 dt_c.

B) Discontinuous initial data (a square step), f = 0, u = 0 on the boundary,
   large time step: Crank-Nicolson (A-stable but not L-stable) damps high
   frequencies very badly -> wrong oscillations. Backward Euler (L-stable) stays smooth.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI

from common import FIGS, RESULTS, HeatConfig, solve_heat, u_exact_numpy, source_ufl

assert MPI.COMM_WORLD.size == 1, "étude de stabilité : exécution séquentielle"
comm = MPI.COMM_WORLD
KAPPA = 1.0


def boundary_dofs(msh, V):
    msh.topology.create_connectivity(1, 2)
    return fem.locate_dofs_topological(V, 1, mesh.exterior_facet_indices(msh.topology))


def to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())


# ---------------------------------------------------------------- A) CFL
N = 16
msh = mesh.create_unit_square(comm, N, N)
V = fem.functionspace(msh, ("Lagrange", 1))
bdofs = boundary_dofs(msh, V)
interior = np.setdiff1d(np.arange(V.dofmap.index_map.size_local), bdofs)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
Kp = petsc.assemble_matrix(fem.form(KAPPA * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)); Kp.assemble()
Mp = petsc.assemble_matrix(fem.form(u * v * ufl.dx)); Mp.assemble()
K = to_scipy(Kp)[interior][:, interior]
M = to_scipy(Mp)[interior][:, interior]
lam_max = float(spla.eigsh(K, k=1, M=M, which="LA", return_eigenvectors=False)[0])
lam_min = float(spla.eigsh(K, k=1, M=M, sigma=0.0, which="LM", return_eigenvectors=False)[0])
dt_c = 2.0 / lam_max
print(f"lambda_min = {lam_min:.4f}  (exact continu 2 pi^2 = {2*np.pi**2:.4f})")
print(f"lambda_max = {lam_max:.1f}  ->  dt_c = {dt_c:.3e}")

# The unstable mode starts from round-off error (~1e-16) and grows like |1 - dt*lambda_max|^n,
# so we need about 1000 steps to see it.
T_expl = 0.3
cfl_runs = {}
for factor in (0.95, 1.05):
    dt = dt_c * factor
    dt = T_expl / np.ceil(T_expl / dt)  # whole number of steps, always on the right side of dt_c
    norms = []

    def cb(n, t, un):
        norms.append((t, float(np.sqrt(comm.allreduce(
            fem.assemble_scalar(fem.form(un * un * ufl.dx)), op=MPI.SUM)))))

    cfg = HeatConfig(kappa=KAPPA, theta=0.0, dt=dt, T=T_expl)
    solve_heat(V, cfg, u0=u_exact_numpy("time")(0.0), source=source_ufl("time", msh, KAPPA),
               dirichlet=[(bdofs, u_exact_numpy("time"))], callback=cb)
    cfl_runs[factor] = dict(dt=dt, ratio=dt / dt_c, norms=norms)
    print(f"theta=0  dt/dt_c={dt/dt_c:.3f}  ||u_h(T)|| = {norms[-1][1]:.3e}")

# ------------------------------------------------- B) CN oscillations
N2, DT2 = 64, 0.05
msh2 = mesh.create_unit_square(comm, N2, N2)
V2 = fem.functionspace(msh2, ("Lagrange", 1))
bd2 = boundary_dofs(msh2, V2)
zero = lambda t: (lambda x: np.zeros(x.shape[1]))  # noqa: E731
square = lambda x: ((np.abs(x[0] - 0.5) <= 0.25) & (np.abs(x[1] - 0.5) <= 0.25)).astype(float)  # noqa: E731
f0 = lambda t: fem.Constant(msh2, 0.0)  # noqa: E731

coords = V2.tabulate_dof_coordinates()
line = np.where(np.abs(coords[:, 1] - 0.5) < 1e-10)[0]
line = line[np.argsort(coords[line, 0])]
xline = coords[line, 0]

profiles, minima = {}, {}
for theta, name, dt in [(0.5, "Crank-Nicolson", DT2), (1.0, "Euler implicite", DT2),
                        (1.0, "Référence", DT2 / 200)]:
    cfg = HeatConfig(kappa=KAPPA, theta=theta, dt=dt, T=DT2)  # one single large step (CN/BE)
    res = solve_heat(V2, cfg, u0=square, source=f0, dirichlet=[(bd2, zero)])
    profiles[name] = res.u.x.array[line].copy()
    minima[name] = (float(res.u.x.array.min()), float(res.u.x.array.max()))
    print(f"{name:16s} t={DT2}: min={minima[name][0]:+.3e}  max={minima[name][1]:.3e}")

# ------------------------------------------------------------- outputs
summary = dict(N=N, lambda_min=lam_min, lambda_min_exact=2 * np.pi**2, lambda_max=lam_max, dt_c=dt_c,
               cfl={str(k): dict(dt=v["dt"], ratio=v["ratio"], final_norm=v["norms"][-1][1])
                    for k, v in cfl_runs.items()},
               rough_data=dict(N=N2, dt=DT2, min_max=minima))
with open(RESULTS / "stability.json", "w") as fh:
    json.dump(summary, fh, indent=2)
with open(RESULTS / "stability_cfl_history.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ratio", "t", "L2_norm"])
    for v in cfl_runs.values():
        for t, nrm in v["norms"]:
            w.writerow([f"{v['ratio']:.4f}", t, nrm])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for (factor, v), c in zip(cfl_runs.items(), ["#2ca02c", "#d62728"]):
    t, n = np.array(v["norms"]).T
    ax.semilogy(t, n, color=c, label=rf"$\Delta t = {v['ratio']:.2f}\,\Delta t_c$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\|u_h(t)\|_{L^2}$")
ax.set_title(rf"Euler explicite, $\mathbb{{P}}_1$, $N={N}$ ($\Delta t_c={dt_c:.2e}$)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

ax = axes[1]
styles = {"Référence": dict(color="k", lw=2.2, alpha=0.35),
          "Euler implicite": dict(color="#1f77b4", lw=1.4),
          "Crank-Nicolson": dict(color="#d62728", lw=1.4)}
for name, prof in profiles.items():
    ax.plot(xline, prof, label=name, **styles[name])
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel(r"$x$  (coupe $y=0.5$)")
ax.set_ylabel(r"$u_h$")
ax.set_title(rf"Donnée initiale créneau, un pas $\Delta t={DT2}$")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "stability.png", dpi=160)
print("-> results/stability.json, figures/stability.png")
