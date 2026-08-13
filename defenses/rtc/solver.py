"""Strict sub-probability QP and independently verified fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class LinearBudget:
    name: str
    coefficients: np.ndarray
    remaining: float


@dataclass(frozen=True)
class SolverResult:
    weights: np.ndarray
    status: str
    optimized: bool
    max_violation: float
    objective: float


def _arrays(
    nominal: Sequence[float] | np.ndarray,
    caps: Sequence[float] | np.ndarray,
    budgets: Sequence[LinearBudget],
) -> tuple[np.ndarray, np.ndarray, list[LinearBudget]]:
    a = np.asarray(nominal, dtype=np.float64).reshape(-1)
    u = np.asarray(caps, dtype=np.float64).reshape(-1)
    if a.size != u.size:
        raise ValueError("nominal masses and caps must have the same length")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("QP variables require finite positive nominal masses")
    if not np.all(np.isfinite(u)) or np.any(u < 0.0) or np.any(u > a):
        raise ValueError("QP caps must be finite and satisfy 0 <= cap <= nominal")
    checked: list[LinearBudget] = []
    for budget in budgets:
        coefficients = np.asarray(budget.coefficients, dtype=np.float64).reshape(-1)
        remaining = max(0.0, float(budget.remaining))
        if coefficients.size != a.size:
            raise ValueError(f"budget {budget.name!r} has the wrong coefficient count")
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
            raise ValueError(f"budget {budget.name!r} coefficients must be non-negative")
        if not np.isfinite(remaining):
            raise ValueError(f"budget {budget.name!r} remaining value must be finite")
        checked.append(LinearBudget(str(budget.name), coefficients, remaining))
    return a, u, checked


def feasible_scale(
    caps: Sequence[float] | np.ndarray,
    budgets: Sequence[LinearBudget],
) -> np.ndarray:
    """Construct the design-specified nonzero feasible warm start."""

    u = np.asarray(caps, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("caps must be finite and non-negative")
    rho = 1.0
    total = float(u.sum(dtype=np.float64))
    if total > 0.0:
        rho = min(rho, 1.0 / total)
    for budget in budgets:
        coefficients = np.asarray(budget.coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size != u.size:
            raise ValueError(f"budget {budget.name!r} has the wrong coefficient count")
        use = float(np.dot(coefficients, u))
        remaining = max(0.0, float(budget.remaining))
        if use > remaining:
            rho = min(rho, remaining / use)
    return np.asarray(max(0.0, rho) * u, dtype=np.float64)


def max_violation(
    weights: Sequence[float] | np.ndarray,
    caps: Sequence[float] | np.ndarray,
    budgets: Sequence[LinearBudget],
) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    u = np.asarray(caps, dtype=np.float64).reshape(-1)
    if w.size != u.size or not np.all(np.isfinite(w)):
        return float("inf")
    violations = [
        float(np.max(-w, initial=0.0)),
        float(np.max(w - u, initial=0.0)),
        max(0.0, float(w.sum(dtype=np.float64)) - 1.0),
    ]
    for budget in budgets:
        coefficients = np.asarray(budget.coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size != w.size:
            return float("inf")
        violations.append(
            max(0.0, float(np.dot(coefficients, w)) - max(0.0, budget.remaining))
        )
    return max(violations, default=0.0)


def solve_subprobability_qp(
    nominal: Sequence[float] | np.ndarray,
    caps: Sequence[float] | np.ndarray,
    budgets: Sequence[LinearBudget],
    *,
    risk: Sequence[float] | np.ndarray | None = None,
    risk_lambda: float = 0.0,
    tolerance: float = 1e-8,
    ftol: float = 1e-10,
    maxiter: int = 100,
) -> SolverResult:
    """Solve the perspective-weighted QP, then verify independently."""

    a, u, checked = _arrays(nominal, caps, budgets)
    risk_array = (
        np.zeros_like(a)
        if risk is None
        else np.asarray(risk, dtype=np.float64).reshape(-1)
    )
    if (
        risk_array.size != a.size
        or not np.all(np.isfinite(risk_array))
        or np.any(risk_array < 0.0)
        or not np.isfinite(risk_lambda)
        or risk_lambda < 0.0
    ):
        raise ValueError("risk inputs must be finite, non-negative, and aligned")

    warm = feasible_scale(u, checked)
    warm_violation = max_violation(warm, u, checked)

    def objective(weights: np.ndarray) -> float:
        difference = weights - a
        return float(
            0.5 * np.dot(difference / a, difference)
            + float(risk_lambda) * np.dot(risk_array, weights)
        )

    def gradient(weights: np.ndarray) -> np.ndarray:
        return (weights - a) / a + float(risk_lambda) * risk_array

    constraints: list[dict[str, object]] = [
        {
            "type": "ineq",
            "fun": lambda weights: 1.0 - float(np.sum(weights, dtype=np.float64)),
            "jac": lambda weights: -np.ones_like(weights, dtype=np.float64),
        }
    ]
    for budget in checked:
        coefficients = budget.coefficients.copy()
        remaining = float(budget.remaining)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, g=coefficients, b=remaining: b
                - float(np.dot(g, weights)),
                "jac": lambda weights, g=coefficients: -g,
            }
        )

    try:
        optimized = minimize(
            objective,
            warm,
            method="SLSQP",
            jac=gradient,
            bounds=[(0.0, float(cap)) for cap in u],
            constraints=constraints,
            options={"ftol": float(ftol), "maxiter": int(maxiter), "disp": False},
        )
        candidate = np.asarray(optimized.x, dtype=np.float64)
        violation = max_violation(candidate, u, checked)
        if violation <= tolerance:
            return SolverResult(
                weights=candidate,
                status="optimized",
                optimized=True,
                max_violation=violation,
                objective=objective(candidate),
            )
    except Exception:
        pass

    if warm_violation <= tolerance:
        return SolverResult(
            weights=warm,
            status="feasible_scale",
            optimized=False,
            max_violation=warm_violation,
            objective=objective(warm),
        )
    zero = np.zeros_like(a)
    return SolverResult(
        weights=zero,
        status="zero_fallback",
        optimized=False,
        max_violation=max_violation(zero, u, checked),
        objective=objective(zero),
    )
