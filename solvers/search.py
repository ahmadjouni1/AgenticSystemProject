"""How the replica runs the teammate's nested solver: their search, a time limit, then a polish.

``Synthesis`` subclasses ``NestedSynthesis`` and changes none of its methods' maths. It adds:

1. **A deadline.** Once it passes, the next new geometry evaluation stops the outer search,
   and the best design checked so far is kept (with the solver's own rule: the lowest
   error among feasible designs, else the least violating one).
2. **A polish** (``polish``): one SLSQP solve over the design and the joint path together,
   from the nested result, with the same minimax form (minimize z subject to every
   waypoint error <= z) and the same constraint margins (``hard_margins``).

**Why the polish.** On our spatial 3R example (``nested/examples/spatial_3r.json``) the outer
COBYLA search stops 12-17 mm from the known answer, from every start we tried. At that
point several waypoint errors are equal and the maximum has a corner: a step along any
single design axis raises it, and only a narrow combined direction lowers it, which
COBYLA's linear models do not find (its trust region shrinks to its lower bound). The
polish, smooth in all variables at once, then reaches the known design (error < 1e-6 m)
in about 1-2 s. Results report both errors, before and after the polish.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from solvers.nested import InnerResult, NestedSynthesis


class _OutOfTime(Exception):
    pass


class _GoodEnough(Exception):
    pass


class Synthesis(NestedSynthesis):
    def __init__(self, *args: Any, deadline: float | None = None,
                 stop_error: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.deadline = deadline  # time.perf_counter() value; None = no limit
        self.stop_error = stop_error  # meters: a feasible design this close ends the search

    def _late(self) -> bool:
        return self.deadline is not None and time.perf_counter() > self.deadline

    def evaluate_geometry(self, u: np.ndarray) -> InnerResult:
        key = tuple(np.clip(np.asarray(u, float), 0, 1))
        if self.cache and key not in self.cache and self._late():
            raise _OutOfTime
        result = super().evaluate_geometry(u)
        if result.feasible and result.max_error <= self.stop_error:
            raise _GoodEnough
        return result

    def best_checked(self) -> InnerResult:
        """The solver's own end rule, over every geometry evaluated so far."""
        feasible = [r for r in self.cache.values() if r.feasible]
        if feasible:
            return min(feasible, key=lambda r: r.max_error)
        return min(self.cache.values(), key=lambda r: (-r.min_margin, r.max_error))

    def run(self, polish: bool = True, polish_iterations: int = 200) -> tuple[InnerResult,
                                                                              dict[str, Any]]:
        """The nested search, then the polish. Returns the best result and what happened."""
        started = time.perf_counter()
        try:
            _best, outer = self.solve(verbose=False)
            outer_message = str(outer.message)
        except _OutOfTime:
            outer_message = "stopped at the time limit"
        except _GoodEnough:
            outer_message = "stopped: a design within a tenth of the tolerance"
        best = self.best_checked()
        stats: dict[str, Any] = {
            "outer": outer_message, "geometry_evaluations": len(self.cache),
            "nested_max_error": best.max_error, "nested_feasible": best.feasible,
            "nested_seconds": round(time.perf_counter() - started, 3)}
        done = best.feasible and best.max_error <= self.stop_error
        if polish and not done and not self._late():
            started = time.perf_counter()
            polished, iterations = self.polish(best, polish_iterations)
            better = polished.feasible and (not best.feasible
                                            or polished.max_error < best.max_error)
            stats.update(polish_iterations=iterations, polish_max_error=polished.max_error,
                         polish_used=better,
                         polish_seconds=round(time.perf_counter() - started, 3))
            if better:
                best = polished
        return best, stats

    def polish(self, start: InnerResult, iterations: int = 200) -> tuple[InnerResult, int]:
        """SLSQP over (design, interior joint path, z) from ``start``. Stops at the deadline
        and returns the best feasible point it evaluated."""
        task, n_design = self.task, len(self.lower)
        seen: dict[str, Any] = {"best": None, "calls": 0}

        def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = np.clip(x[:n_design], 0, 1)
            return self.lower + self.range * u, self.decode_q(np.clip(x[n_design:-1], 0, 1))

        def constraints(x: np.ndarray) -> np.ndarray:
            if self._late():
                raise _OutOfTime
            seen["calls"] += 1
            design, q = split(x)
            fk = self.model.fk(design, q)
            errors = np.sqrt(np.sum(((fk[0] - task.positions) / task.length_scale) ** 2,
                                    axis=1) + 1e-20)
            margins = self.hard_margins(design, q, fk)
            if margins.min() >= -self.settings.feasibility_tolerance:
                worst = float(errors.max())
                if seen["best"] is None or worst < seen["best"][0]:
                    seen["best"] = (worst, design.copy(), q.copy())
                if worst * task.length_scale <= self.stop_error:
                    raise _GoodEnough
            return np.r_[x[-1] - errors, margins]

        y0 = ((start.q[1:-1] - self.model.lower) / self.q_range).ravel()
        u0 = (start.design - self.lower) / self.range
        x0 = np.r_[u0, y0, start.max_error / task.length_scale + 1e-8]
        gradient = np.r_[np.zeros(len(x0) - 1), 1.0]
        outcome, message, nit = None, "", 0
        try:
            outcome = minimize(lambda x: x[-1], x0, jac=lambda x: gradient, method="SLSQP",
                               bounds=[(0, 1)] * (len(x0) - 1) + [(0, None)],
                               constraints=[{"type": "ineq", "fun": constraints}],
                               options={"maxiter": iterations, "ftol": 1e-12})
            message, nit = f"polish: {outcome.message}", int(outcome.nit)
            constraints(outcome.x)  # make sure the final point is considered
        except _OutOfTime:
            message = "polish: stopped at the time limit"
        except _GoodEnough:
            message = "polish: stopped within a tenth of the tolerance"
        if seen["best"] is None:
            return InnerResult(start.design, start.q, start.max_error, start.min_margin,
                               start.feasible, False, message + "; no feasible point"), nit
        _worst, design, q = seen["best"]
        return self._result(design, q, outcome is not None and bool(outcome.success),
                            message), nit
