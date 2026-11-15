"""
General finite element solver for the heat equation (FEniCSx / DOLFINx 0.11).

    du/dt - div(kappa grad u) + beta (u - u_amb) = f   in Omega x (0, T]
    u = g                                    on Gamma_D
    -kappa grad u . n = h_c (u - u_amb)      on Gamma_R   (Robin, optional)
    kappa grad u . n = 0                     elsewhere     (natural Neumann)

Time discretisation: theta scheme
    theta = 1   -> backward Euler    (order 1, L-stable)
    theta = 1/2 -> Crank-Nicolson    (order 2, A-stable)
    theta = 0   -> forward Euler     (order 1, stable only for small dt)

The time step is constant, so the matrix A = M + theta*dt*K (+ Robin) is
assembled and factorised only ONCE. Only the right-hand side is assembled
again at each step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc


@dataclass
class HeatConfig:
    kappa: float = 1.0
    theta: float = 1.0
    dt: float = 1e-2
    T: float = 1.0
    # Robin (convection): -kappa du/dn = h_c (u - u_amb)
    h_conv: float = 0.0
    u_amb: float = 0.0
    # Volume exchange term beta (u - u_amb) (e.g. heat loss through the faces of a thin plate)
    beta: float = 0.0
    solver: str = "lu"  # "lu" (MUMPS) or "cg" (CG + Hypre BoomerAMG)
    quiet: bool = True


@dataclass
class HeatResult:
    u: fem.Function
    times: list = field(default_factory=list)
    n_steps: int = 0
    ksp_its: list = field(default_factory=list)


def solve_heat(
    V: fem.FunctionSpace,
    cfg: HeatConfig,
    u0: Callable | fem.Function,
    source: Callable[[fem.Constant], ufl.core.expr.Expr],
    dirichlet: Sequence[tuple[np.ndarray, Callable[[float], Callable]]] = (),
    robin_ds: Optional[ufl.Measure] = None,
    callback: Optional[Callable[[int, float, fem.Function], None]] = None,
) -> HeatResult:
    """
    Parameters
    ----------
    V         : scalar finite element space.
    cfg       : physical and numerical parameters.
    u0        : initial condition (numpy callable x -> values, or a Function).
    source    : function t (fem.Constant) -> UFL expression of f(x, t).
    dirichlet : list of (dofs, g) where g(t) returns a numpy callable x -> values.
    robin_ds  : boundary measure where the Robin condition is applied.
    callback  : called after each step: callback(n, t, u_n).
    """
    msh = V.mesh
    dtype = PETSc.ScalarType
    kappa = fem.Constant(msh, dtype(cfg.kappa))
    dt = fem.Constant(msh, dtype(cfg.dt))
    th = fem.Constant(msh, dtype(cfg.theta))
    h_c = fem.Constant(msh, dtype(cfg.h_conv))
    u_amb = fem.Constant(msh, dtype(cfg.u_amb))

    # Times t^n and t^{n+1}: the source term is a theta-weighted average of the two
    t_old = fem.Constant(msh, dtype(0.0))
    t_new = fem.Constant(msh, dtype(cfg.dt))

    u_n = fem.Function(V, name="u")
    if isinstance(u0, fem.Function):
        u_n.x.array[:] = u0.x.array
    else:
        u_n.interpolate(u0)
    u_h = fem.Function(V, name="u")
    u_h.x.array[:] = u_n.x.array

    # Dirichlet conditions (they can change with time)
    bcs, g_funcs, g_makers = [], [], []
    for dofs, g_of_t in dirichlet:
        g = fem.Function(V)
        g.interpolate(g_of_t(cfg.dt))
        bcs.append(fem.dirichletbc(g, dofs))
        g_funcs.append(g)
        g_makers.append(g_of_t)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.dx

    # Bilinear form: (u, v) + theta dt [ kappa (grad u, grad v) + h_c <u, v>_R ]
    a = u * v * dx + th * dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    # Linear form: (u_n, v) - (1-theta) dt kappa (grad u_n, grad v) + dt (f^{n+theta}, v)
    f_theta = th * source(t_new) + (1 - th) * source(t_old)
    L = (
        u_n * v * dx
        - (1 - th) * dt * kappa * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * dx
        + dt * f_theta * v * dx
    )
    if robin_ds is not None and cfg.h_conv != 0.0:
        a += th * dt * h_c * u * v * robin_ds
        L += -(1 - th) * dt * h_c * u_n * v * robin_ds + dt * h_c * u_amb * v * robin_ds

    if cfg.beta != 0.0:
        beta = fem.Constant(msh, dtype(cfg.beta))
        a += th * dt * beta * u * v * dx
        L += -(1 - th) * dt * beta * u_n * v * dx + dt * beta * u_amb * v * dx

    a_form, L_form = fem.form(a), fem.form(L)

    # Assemble the matrix only once
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(fem.extract_function_spaces(L_form))

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    if cfg.solver == "lu":
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
    else:
        ksp.setType("cg")
        ksp.getPC().setType("hypre")
        ksp.getPC().setHYPREType("boomeramg")
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setTolerances(rtol=1e-12, atol=1e-14)
    ksp.setFromOptions()

    res = HeatResult(u=u_h)
    n_steps = int(round(cfg.T / cfg.dt))
    t = 0.0
    res.times.append(t)
    if callback is not None:
        callback(0, t, u_n)

    for n in range(1, n_steps + 1):
        t_old.value = t
        t = n * cfg.dt
        t_new.value = t
        for g, g_of_t in zip(g_funcs, g_makers):
            g.interpolate(g_of_t(t))

        with b.localForm() as b_loc:
            b_loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        res.ksp_its.append(ksp.getIterationNumber())

        u_n.x.array[:] = u_h.x.array
        res.times.append(t)
        if callback is not None:
            callback(n, t, u_n)

    res.n_steps = n_steps
    ksp.destroy()
    A.destroy()
    b.destroy()
    return res


def errornorms(u_h: fem.Function, u_exact_ufl, degree_raise: int = 3) -> tuple[float, float]:
    """Errors ||u - u_h||_{L2} and |u - u_h|_{H1}, computed with a higher quadrature degree."""
    msh = u_h.function_space.mesh
    deg = u_h.function_space.ufl_element().embedded_superdegree + degree_raise
    dxq = ufl.dx(metadata={"quadrature_degree": deg})
    e = u_h - u_exact_ufl
    l2 = fem.assemble_scalar(fem.form(e**2 * dxq))
    h1 = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dxq))
    l2 = msh.comm.allreduce(l2, op=MPI.SUM)
    h1 = msh.comm.allreduce(h1, op=MPI.SUM)
    return float(np.sqrt(l2)), float(np.sqrt(h1))
