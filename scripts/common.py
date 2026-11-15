"""Shared helpers for the verification studies (manufactured problem on the unit square)."""
import sys
from pathlib import Path

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

from heat import HeatConfig, errornorms, solve_heat  # noqa: E402
from heat.manufactured import source_ufl, u_exact_numpy, u_exact_ufl  # noqa: E402


def run_manufactured(case: str, N: int, degree: int, dt: float, T: float, theta: float,
                     kappa: float = 1.0, solver: str = "lu"):
    """Solve the manufactured problem and return (err_L2, err_H1, ndofs) at t = T."""
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    bfacets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bfacets)

    cfg = HeatConfig(kappa=kappa, theta=theta, dt=dt, T=T, solver=solver)
    res = solve_heat(
        V, cfg,
        u0=u_exact_numpy(case)(0.0),
        source=source_ufl(case, msh, kappa),
        dirichlet=[(bdofs, u_exact_numpy(case))],
    )
    t_final = fem.Constant(msh, res.times[-1])
    eL2, eH1 = errornorms(res.u, u_exact_ufl(case, msh, t_final))
    ndofs = V.dofmap.index_map.size_global
    return eL2, eH1, ndofs


def rates(errs, hs):
    errs, hs = np.asarray(errs), np.asarray(hs)
    r = np.log(errs[1:] / errs[:-1]) / np.log(hs[1:] / hs[:-1])
    return np.concatenate([[np.nan], r])
