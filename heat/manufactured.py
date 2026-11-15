"""
Manufactured solutions used to check the code (Method of Manufactured Solutions).

Case "space"  (isolate the space error)
    u_ex = (1 + t) sin(pi x) sin(pi y) + x + y
    f    = [1 + 2 kappa pi^2 (1 + t)] sin(pi x) sin(pi y)
  u_ex is linear in t, so the theta scheme (with f taken as a theta-weighted
  average) is EXACT in time. The measured error comes only from space.

Case "time"   (isolate the time error)
    u_ex = exp(-t) [ sin(pi x) sin(pi y) + x + y ]
    f    = exp(-t) [ (2 kappa pi^2 - 1) sin(pi x) sin(pi y) - (x + y) ]
  u_ex is not a polynomial in t, so the time error is not zero. We measure it
  on a fine, high-order mesh.

In both cases the linear part (x + y) makes the Dirichlet condition
non-zero (and time-dependent in the "time" case).
"""
import numpy as np
import ufl


def u_exact_numpy(case: str):
    """Return t -> (numpy callable x -> u_ex(x, t)), for interpolate()."""
    def at_time(t: float):
        def g(x):
            s = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
            if case == "space":
                return (1.0 + t) * s + x[0] + x[1]
            return np.exp(-t) * (s + x[0] + x[1])
        return g
    return at_time


def u_exact_ufl(case: str, msh, t):
    x = ufl.SpatialCoordinate(msh)
    s = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    if case == "space":
        return (1.0 + t) * s + x[0] + x[1]
    return ufl.exp(-t) * (s + x[0] + x[1])


def source_ufl(case: str, msh, kappa: float):
    """Return t -> f(x, t) as UFL (t is a fem.Constant)."""
    x = ufl.SpatialCoordinate(msh)
    s = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    def f(t):
        if case == "space":
            return (1.0 + 2.0 * kappa * ufl.pi**2 * (1.0 + t)) * s
        return ufl.exp(-t) * ((2.0 * kappa * ufl.pi**2 - 1.0) * s - (x[0] + x[1]))
    return f
