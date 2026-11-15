"""
Application: thin steel plate heated by a moving Gaussian laser source.

2D model (plate of thickness e, temperature is the same through the thickness):

    rho c dT/dt - div(k grad T) = q(x, y, t)                    in Omega = (0,Lx) x (0,Ly)
    -k grad T . n = h (T - T_amb)                               on the whole boundary (Robin)

    After integrating through the thickness, convection on the two faces of the
    plate gives a volume term (2 h / e)(T - T_amb) added to the left-hand side.

    q = (eta P / e) * 1/(2 pi s^2) * exp( -|x - x_s(t)|^2 / (2 s^2) ),
    x_s(t) = (x0 + v t, Ly/2) for 0 <= t <= t_on, then the source is switched off.

Dividing by rho c: alpha = k/(rho c), f = q/(rho c), h_c = h/(rho c) -> same form
as the general solver.

Outputs: XDMF (ParaView), temperature probes, energy balance, PNG snapshots.
Runs in parallel: mpirun -n 2 python laser.py
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from dolfinx import fem, geometry, io, mesh
from mpi4py import MPI

from common import FIGS, RESULTS, HeatConfig, solve_heat

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=200, help="cells along x (ny = nx/2)")
p.add_argument("--degree", type=int, default=1)
p.add_argument("--dt", type=float, default=0.05)
p.add_argument("--theta", type=float, default=0.5)
p.add_argument("--solver", default="lu", choices=["lu", "cg"])
p.add_argument("--no-output", action="store_true", help="no XDMF / figures (mesh study)")
p.add_argument("--tag", default="")
args = p.parse_args()

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------- physics (steel)
k, rho, c = 45.0, 7850.0, 460.0           # W/(m K), kg/m^3, J/(kg K)
alpha = k / (rho * c)                      # m^2/s
h, T_amb = 25.0, 20.0                      # W/(m^2 K), degC
e = 2e-3                                   # thickness (m)
P, eta, s = 400.0, 0.5, 1.5e-3             # power (W), absorptivity, Gaussian radius (m)
Lx, Ly = 0.10, 0.05                        # m
x0, v, t_on = 0.02, 4e-3, 15.0             # start (m), speed (m/s), heating time (s)
T_end = 25.0

# ---------------------------------------------------------------- mesh / space
nx = args.nx
msh = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([Lx, Ly])], [nx, nx // 2],
                            mesh.CellType.triangle)
V = fem.functionspace(msh, ("Lagrange", args.degree))
xs = ufl.SpatialCoordinate(msh)
on = fem.Constant(msh, 1.0)  # laser on/off switch (updated in the callback)


def source(t):
    """f = q / (rho c); t is a fem.Constant (time t^n or t^{n+1})."""
    xc = x0 + v * ufl.min_value(t, t_on)
    r2 = (xs[0] - xc) ** 2 + (xs[1] - Ly / 2) ** 2
    q = eta * P / e / (2 * np.pi * s**2) * ufl.exp(-r2 / (2 * s**2))
    # sharp switch-off at t_on: 1 if t <= t_on, else 0
    return ufl.conditional(ufl.le(t, t_on + 1e-12), 1.0, 0.0) * q / (rho * c)


cfg = HeatConfig(kappa=alpha, theta=args.theta, dt=args.dt, T=T_end,
                 h_conv=h / (rho * c), u_amb=T_amb, beta=2 * h / (e * rho * c),
                 solver=args.solver)

# ---------------------------------------------------------------- probes
probes = np.array([[0.05, Ly / 2, 0.0],          # on the laser path
                   [0.05, Ly / 2 + 5e-3, 0.0],   # 5 mm from the path
                   [0.05, Ly / 2 + 10e-3, 0.0]]) # 10 mm from the path
tree = geometry.bb_tree(msh, msh.topology.dim)
cand = geometry.compute_collisions_points(tree, probes)
coll = geometry.compute_colliding_cells(msh, cand, probes)
probe_cells = [coll.links(i)[0] if len(coll.links(i)) > 0 else -1 for i in range(len(probes))]


def eval_probes(u):
    vals = np.full(len(probes), -np.inf)
    for i, cell in enumerate(probe_cells):
        if cell >= 0:
            vals[i] = u.eval(probes[i:i + 1], np.array([cell], dtype=np.int32))[0]
    out = np.empty_like(vals)
    comm.Allreduce(vals, out, op=MPI.MAX)  # buffer (numpy) version, not the pickle-based allreduce
    return out


# ---------------------------------------------------------------- energy balance
# Energy (J) = rho c e * int T dx. With test function v = 1, the scheme conserves exactly:
#   int (T^{n+1} - T^n) = dt int f^{n+theta} - dt int_dOmega h_c (T^{n+theta} - T_amb)
def integral(expr, measure=ufl.dx):
    return comm.allreduce(fem.assemble_scalar(fem.form(expr * measure)), op=MPI.SUM)


one = fem.Constant(msh, 1.0)
perim, area = integral(one, ufl.ds), integral(one)
t_c = fem.Constant(msh, 0.0)
# int f dx is computed as int f * v dx with v = 1 in V: same quadrature degree as
# the solver right-hand side (otherwise a poorly resolved Gaussian gives a quadrature gap)
one_h = fem.Function(V)
one_h.x.array[:] = 1.0
f_form = fem.form(source(t_c) * one_h * ufl.dx)
history = dict(t=[], probes=[], Tmax=[], energy=[], energy_in=[], energy_out=[])
state = dict(prev=None, Ein=0.0, Eout=0.0)
snap_times = [3.75, 7.5, 11.25, 15.0, 20.0, 25.0]
snapshots = {}

coords_owned = V.tabulate_dof_coordinates()[: V.dofmap.index_map.size_local, :2]
coords_all = comm.gather(coords_owned, root=0)
if rank == 0:
    coords_all = np.vstack(coords_all)

xdmf = None
if not args.no_output:
    xdmf = io.XDMFFile(comm, str(RESULTS / f"laser{args.tag}.xdmf"), "w")
    xdmf.write_mesh(msh)


def flux_in(t):
    t_c.value = t
    return comm.allreduce(fem.assemble_scalar(f_form), op=MPI.SUM)


post_forms = {}  # post-processing forms, compiled once (not at every step)


def callback(n, t, u):
    th = args.theta
    if not post_forms:
        post_forms["U"] = fem.form(u * ufl.dx)
        post_forms["Ub"] = fem.form(u * ufl.ds)
    U = comm.allreduce(fem.assemble_scalar(post_forms["U"]), op=MPI.SUM)
    Ub = comm.allreduce(fem.assemble_scalar(post_forms["Ub"]), op=MPI.SUM)
    # current heat loss: edges (Robin) + faces (volume term)
    Fout = cfg.h_conv * (Ub - T_amb * perim) + cfg.beta * (U - T_amb * area)
    Fin = flux_in(t)
    if state["prev"] is not None:
        Fin_prev, Fout_prev = state["prev"]
        state["Ein"] += args.dt * (th * Fin + (1 - th) * Fin_prev)
        state["Eout"] += args.dt * (th * Fout + (1 - th) * Fout_prev)
    else:
        state["U0"] = U
    state["prev"] = (Fin, Fout)

    Tmax = comm.allreduce(u.x.array[: V.dofmap.index_map.size_local].max(), op=MPI.MAX)
    history["t"].append(t)
    history["probes"].append(eval_probes(u).tolist())
    history["Tmax"].append(float(Tmax))
    rc_e = rho * c * e
    history["energy"].append(rc_e * (U - state["U0"]))
    history["energy_in"].append(rc_e * state["Ein"])
    history["energy_out"].append(rc_e * state["Eout"])

    if xdmf is not None and n % 10 == 0:
        xdmf.write_function(u, t)
    if not args.no_output and any(abs(t - ts) < 1e-9 for ts in snap_times):
        vals = comm.gather(u.x.array[: V.dofmap.index_map.size_local].copy(), root=0)
        if rank == 0:
            snapshots[round(t, 6)] = np.concatenate(vals)


tic = time.perf_counter()
res = solve_heat(V, cfg, u0=lambda x: np.full(x.shape[1], T_amb), source=source,
                 dirichlet=[], robin_ds=ufl.ds, callback=callback)
wall = time.perf_counter() - tic
if xdmf is not None:
    xdmf.close()

# ---------------------------------------------------------------- post-processing
ndofs = V.dofmap.index_map.size_global
E, Ein, Eout = (np.array(history[k_]) for k_ in ("energy", "energy_in", "energy_out"))
balance_err = float(np.max(np.abs(E - (Ein - Eout))) / max(Ein.max(), 1e-30))
Tmax = np.array(history["Tmax"])
summary = dict(nx=nx, degree=args.degree, dt=args.dt, theta=args.theta, ndofs=ndofs,
               mpi_size=comm.size, solver=args.solver, wall_time_s=wall, n_steps=res.n_steps,
               Tmax_peak=float(Tmax.max()), t_Tmax_peak=float(history["t"][int(Tmax.argmax())]),
               probe_peaks=np.max(np.array(history["probes"]), axis=0).tolist(),
               energy_in_J=float(Ein[-1]), energy_out_J=float(Eout[-1]), energy_stored_J=float(E[-1]),
               energy_balance_rel_err=balance_err,
               mean_ksp_its=float(np.mean(res.ksp_its)))
if rank == 0:
    print(json.dumps(summary, indent=2))
    with open(RESULTS / f"laser_summary{args.tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

if rank == 0 and not args.no_output:
    t = np.array(history["t"])
    pr = np.array(history["probes"])
    np.savetxt(RESULTS / "laser_history.csv",
               np.column_stack([t, Tmax, pr, E, Ein, Eout]), delimiter=",",
               header="t,Tmax,T_probe0mm,T_probe5mm,T_probe10mm,E_stored,E_in,E_out", comments="")

    # Snapshots
    tri = mtri.Triangulation(coords_all[:, 0] * 1e3, coords_all[:, 1] * 1e3)
    vmax = max(sn.max() for sn in snapshots.values())
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), constrained_layout=True)
    levels = np.linspace(T_amb, vmax, 41)
    for ax, (ts, vals) in zip(axes.flat, sorted(snapshots.items())):
        cs = ax.tricontourf(tri, vals, levels=levels, cmap="inferno", extend="both")
        ax.tricontour(tri, vals, levels=[100, 200, 400, 600], colors="w", linewidths=0.5, alpha=0.6)
        if ts <= t_on:
            ax.plot((x0 + v * ts) * 1e3, Ly / 2 * 1e3, "c+", ms=10, mew=1.5)
        ax.plot(probes[:, 0] * 1e3, probes[:, 1] * 1e3, "o", mfc="none", mec="c", ms=4)
        ax.set_aspect("equal")
        ax.set_title(rf"$t = {ts:g}$ s" + ("  (laser coupé)" if ts > t_on else ""))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.colorbar(cs, ax=axes, shrink=0.8, label=r"$T$ (°C)")
    fig.savefig(FIGS / "laser_snapshots.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(t, Tmax, "k", lw=1.6, label=r"$\max_\Omega T$")
    for j, lab in enumerate(["sonde 0 mm", "sonde 5 mm", "sonde 10 mm"]):
        ax.plot(t, pr[:, j], label=lab)
    ax.axvline(t_on, color="gray", ls=":", label="coupure laser")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T (°C)")
    ax.set_title("Températures")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    ax.plot(t, Ein, label=r"énergie injectée $E_{in}$")
    ax.plot(t, Eout, label=r"pertes convectives $E_{out}$ (faces + bords)")
    ax.plot(t, E, "k--", label=r"énergie stockée $\Delta E$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("J")
    ax.set_title(rf"Bilan d'énergie (erreur relative max $= {balance_err:.1e}$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "laser_history.png", dpi=160)
    print("-> results/laser.xdmf, figures/laser_snapshots.png, figures/laser_history.png")
