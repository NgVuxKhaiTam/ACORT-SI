"""Response-parametric weighted-Lasso homotopy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


class HomotopyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PathSegment:
    z_left: float
    z_right: float
    beta_intercept: np.ndarray
    beta_slope: np.ndarray

    def evaluate(self, z):
        return self.beta_intercept + self.beta_slope * z


def _solve_active_system(active_design, rhs):
    gram = active_design.T @ active_design
    return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _polish_solution(X, y, penalty_weights, beta):
    active = np.flatnonzero(beta != 0.0)
    polished = np.zeros_like(beta)
    if active.size == 0:
        return polished
    signs = np.sign(beta[active])
    active_design = X[:, active]
    rhs = active_design.T @ y - penalty_weights[active] * signs
    polished[active] = _solve_active_system(active_design, rhs)
    return polished


def _critical_cone_direction(X, y, beta, response_direction, penalty_weights, direction_tol, cone_tol):
    correlation = X.T @ (y - X @ beta)
    free = np.flatnonzero(beta != 0.0)
    free_mask = np.zeros(beta.size, dtype=bool)
    free_mask[free] = True
    boundary = np.flatnonzero((~free_mask) & (np.abs(correlation) >= penalty_weights) )
    derivative = np.zeros(beta.size, dtype=float)
    if free.size + boundary.size == 0:
        return derivative, X.T @ response_direction, free

    boundary_sign = np.sign(correlation[boundary])
    boundary_design = X[:, boundary] * boundary_sign[np.newaxis, :]
    cone_matrix = np.column_stack([X[:, free], boundary_design])
    lower = np.concatenate([np.full(free.size, -np.inf), np.zeros(boundary.size)] )
    upper = np.full(free.size + boundary.size, np.inf)
    result = lsq_linear(cone_matrix, response_direction, bounds=(lower, upper), method="bvls", tol=cone_tol, max_iter=max(100, 10 * cone_matrix.shape[1]))
    derivative[free] = result.x[: free.size]
    boundary_magnitude = result.x[free.size :]
    derivative[boundary] = boundary_sign * boundary_magnitude
    entering = boundary[boundary_magnitude > direction_tol]
    outgoing_active = np.sort(np.concatenate([free, entering]))
    correlation_slope = X.T @ (response_direction - X @ derivative)
    return derivative, correlation_slope, outgoing_active


def _next_breakpoint(beta, derivative, correlation, correlation_slope, penalty_weights, outgoing_active, remaining, direction_tol, step_tol):
    candidates = []
    active_mask = np.zeros(beta.size, dtype=bool)
    active_mask[outgoing_active] = True

    for idx in outgoing_active:
        if beta[idx] == 0.0 or abs(derivative[idx]) <= direction_tol:
            continue
        step = -beta[idx] / derivative[idx]
        if step > 0.0:
            candidates.append((float(step), int(idx)))

    for idx in np.flatnonzero(~active_mask):
        slope = correlation_slope[idx]
        if slope > direction_tol:
            step = (penalty_weights[idx] - correlation[idx]) / slope
        elif slope < -direction_tol:
            step = (-penalty_weights[idx] - correlation[idx]) / slope
        else:
            continue
        if step > 0.0:
            candidates.append((float(step), None))

    if not candidates:
        return remaining, np.empty(0, dtype=int)

    raw_step = min(step for step, _ in candidates)
    if raw_step >= remaining - step_tol:
        return remaining, np.empty(0, dtype=int)
    step = max(raw_step, step_tol)
    leaving = [idx for value, idx in candidates if idx is not None and abs(value - raw_step) <= step_tol ]
    return step, np.asarray(leaving, dtype=int)


def _trace_ray(X, y_start, beta_start, response_direction, penalty_weights, t_limit, direction_tol=2e-10, step_tol=2e-10, cone_tol=2e-12, max_breakpoints=100_000):
    current_t = 0.0
    current_y = np.asarray(y_start, dtype=float).copy()
    beta = _polish_solution(X, current_y, penalty_weights, np.asarray(beta_start, dtype=float).copy())
    segments = []

    for _ in range(max_breakpoints):
        remaining = t_limit - current_t
        if remaining <= step_tol * max(1.0, t_limit):
            break
        derivative, correlation_slope, outgoing_active = (_critical_cone_direction(X, current_y, beta, response_direction, penalty_weights, direction_tol, cone_tol) )
        correlation = X.T @ (current_y - X @ beta)
        step, leaving = _next_breakpoint(beta, derivative, correlation, correlation_slope, penalty_weights, outgoing_active, remaining, direction_tol, step_tol)
        segments.append((current_t, current_t + step, beta.copy(), derivative.copy()) )
        current_t += step
        current_y = np.asarray(y_start) + response_direction * current_t
        beta = beta + derivative * step
        beta[leaving] = 0.0
        if current_t >= t_limit - step_tol:
            break
        beta = _polish_solution(X, current_y, penalty_weights, beta)
    else:
        raise HomotopyError(f"The response path exceeded {max_breakpoints} breakpoints" )

    return segments, beta


def compute_solution_path(X, y_anchor, response_slope, penalty_weights, z_anchor, z_min, z_max, beta_anchor=None, **trace_options):
    """Return affine weighted-Lasso path segments and the anchor solution."""
    X = np.asfortranarray(np.asarray(X, dtype=float))
    y_anchor = np.asarray(y_anchor, dtype=float).reshape(-1)
    response_slope = np.asarray(response_slope, dtype=float).reshape(-1)
    penalty_weights = np.asarray(penalty_weights, dtype=float).reshape(-1)
    if X.ndim != 2 or y_anchor.shape != (X.shape[0],):
        raise ValueError("Incompatible X and y_anchor dimensions")
    if response_slope.shape != y_anchor.shape:
        raise ValueError("response_slope must have the response dimension")
    if penalty_weights.shape != (X.shape[1],):
        raise ValueError("penalty_weights must have the coefficient dimension")
    if np.any(penalty_weights <= 0.0):
        raise ValueError("penalty_weights must be strictly positive")
    if not z_min <= z_anchor <= z_max:
        raise ValueError("z_anchor must belong to [z_min, z_max]")

    if beta_anchor is None:
        _, beta_anchor = _trace_ray(X, np.zeros_like(y_anchor), np.zeros(X.shape[1]), y_anchor, penalty_weights, 1.0, **trace_options)
    else:
        beta_anchor = np.asarray(beta_anchor, dtype=float).reshape(-1)
        if beta_anchor.shape != (X.shape[1],):
            raise ValueError("beta_anchor has the wrong dimension")
        beta_anchor = _polish_solution(X, y_anchor, penalty_weights, beta_anchor)

    left_ray, _ = _trace_ray(X, y_anchor, beta_anchor, -response_slope, penalty_weights, z_anchor - z_min, **trace_options)
    right_ray, _ = _trace_ray(X, y_anchor, beta_anchor, response_slope, penalty_weights, z_max - z_anchor, **trace_options)

    segments = []
    for t_left, t_right, beta_left, derivative in left_ray:
        z_right = z_anchor - t_left
        z_left = z_anchor - t_right
        slope = -derivative
        intercept = beta_left - slope * z_right
        segments.append(PathSegment(z_left, z_right, intercept, slope.copy()))
    for t_left, t_right, beta_left, derivative in right_ray:
        z_left = z_anchor + t_left
        z_right = z_anchor + t_right
        intercept = beta_left - derivative * z_left
        segments.append(PathSegment(z_left, z_right, intercept, derivative.copy()) )
    segments.sort(key=lambda segment: (segment.z_left, segment.z_right))
    return segments, beta_anchor


def solve_weighted_lasso(X, y, penalty_weights, **trace_options):
    """Solve one weighted Lasso through the auxiliary path from zero."""
    X = np.asfortranarray(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).reshape(-1)
    penalty_weights = np.asarray(penalty_weights, dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("Incompatible X and y dimensions")
    if penalty_weights.shape != (X.shape[1],):
        raise ValueError("penalty_weights must have the coefficient dimension")
    if np.any(penalty_weights <= 0.0):
        raise ValueError("penalty_weights must be strictly positive")
    _, beta = _trace_ray(X, np.zeros_like(y), np.zeros(X.shape[1]), y, penalty_weights, 1.0, **trace_options)
    return beta


__all__ = ["HomotopyError", "PathSegment", "compute_solution_path", "solve_weighted_lasso"]
