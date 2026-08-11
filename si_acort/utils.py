"""Shared constructions and selective-inference utilities for SI-ACoRT."""

from __future__ import annotations

import math

import numpy as np
import scipy.special


def construct_active_set(beta):
    """Return the exact nonzero support of a coefficient vector."""
    return np.flatnonzero(np.asarray(beta) != 0.0).tolist()


def construct_folds(n0, T=5, shuffle=False, random_state=0):
    if T <= 0:
        raise ValueError("T must be positive")
    if T > n0:
        raise ValueError("T cannot exceed the target sample size")
    indices = np.arange(n0)
    if shuffle:
        indices = np.random.default_rng(random_state).permutation(indices)
    return [np.asarray(part, dtype=int) for part in np.array_split(indices, T)]


def complement_fold_indices(n0, fold_indices):
    mask = np.ones(n0, dtype=bool)
    mask[np.asarray(fold_indices, dtype=int)] = False
    return np.flatnonzero(mask)


def construct_block_slices(n_list):
    result = []
    start = 0
    for size in n_list:
        result.append(slice(start, start + size))
        start += size
    return result


def construct_X_tilde(X_list, I):
    p = X_list[0].shape[1]
    rows = []
    for block_idx, source_idx in enumerate(I):
        Xk = np.asarray(X_list[source_idx], dtype=float)
        scaled = Xk / math.sqrt(Xk.shape[0])
        blocks = [scaled if idx == block_idx else np.zeros_like(scaled) for idx in range(len(I)) ]
        blocks.append(scaled)
        rows.append(np.hstack(blocks))

    X0 = np.asarray(X_list[-1], dtype=float)
    target_blocks = [np.zeros((X0.shape[0], p)) for _ in I]
    target_blocks.append(X0 / math.sqrt(X0.shape[0]))
    rows.append(np.hstack(target_blocks))
    return np.asfortranarray(np.vstack(rows))


def construct_Y_tilde(Y_list, n_list, I):
    return np.concatenate([np.asarray(Y_list[idx], dtype=float) / math.sqrt(n_list[idx]) for idx in I] + [np.asarray(Y_list[-1], dtype=float) / math.sqrt(n_list[-1])] )


def construct_f(n_list, I):
    selected_sizes = np.asarray(n_list)[I + [-1]]
    return np.repeat(1.0 / np.sqrt(selected_sizes), selected_sizes ).reshape(-1, 1)


def construct_w_tilde(p, lambda_list, I):
    return np.concatenate([np.full(p, lambda_list[idx], dtype=float) for idx in I] + [np.full(p, lambda_list[-1], dtype=float)] )


def merge_intervals(intervals, tol=1e-8):
    if not intervals:
        return []
    ordered = sorted((float(left), float(right)) for left, right in intervals if left <= right + tol )
    if not ordered:
        return []
    merged = [ordered[0]]
    for left, right in ordered[1:]:
        if left <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def intersect_interval_lists(intervals_a, intervals_b, tol=1e-8):
    intervals_a = merge_intervals(intervals_a, tol)
    intervals_b = merge_intervals(intervals_b, tol)
    result = []
    i = j = 0
    while i < len(intervals_a) and j < len(intervals_b):
        left = max(intervals_a[i][0], intervals_b[j][0])
        right = min(intervals_a[i][1], intervals_b[j][1])
        if left <= right + tol:
            result.append((left, right))
        if intervals_a[i][1] < intervals_b[j][1] - tol:
            i += 1
        else:
            j += 1
    return merge_intervals(result, tol)


def solve_quadratic_ineq(A, B, C, tol=1e-12):
    """Solve A*z**2 + B*z + C <= 0 on the real line."""
    A = float(A)
    B = float(B)
    C = float(C)
    if abs(A) <= tol:
        if abs(B) <= tol:
            return [(-np.inf, np.inf)] if C <= tol else []
        root = -C / B
        return [(-np.inf, root)] if B > 0 else [(root, np.inf)]

    discriminant = B * B - 4.0 * A * C
    if discriminant < -tol:
        return [(-np.inf, np.inf)] if A < 0 else []
    if abs(discriminant) <= tol:
        root = -B / (2.0 * A)
        return [(-np.inf, np.inf)] if A < 0 else [(root, root)]

    sqrt_discriminant = math.sqrt(max(discriminant, 0.0))
    root1 = (-B - sqrt_discriminant) / (2.0 * A)
    root2 = (-B + sqrt_discriminant) / (2.0 * A)
    left, right = min(root1, root2), max(root1, root2)
    if A > 0:
        return [(left, right)]
    return [(-np.inf, left), (right, np.inf)]


def point_in_interval_list(value, intervals, tol=1e-10):
    return any(left - tol <= value <= right + tol for left, right in intervals)


def construct_test_statistic(j, X0M, Y, M, n0, n):
    M = list(M)
    if j not in M:
        raise ValueError("j must belong to M")
    if not 0 < n0 <= n:
        raise ValueError("The sample sizes must satisfy 0 < n0 <= n")
    X0M = np.asarray(X0M, dtype=float)
    if X0M.ndim != 2 or X0M.shape != (n0, len(M)):
        raise ValueError("X0M must have shape (n0, len(M)) for the selected target design" )
    ej = np.zeros(len(M))
    ej[M.index(j)] = 1.0
    gram_solution = np.linalg.solve(X0M.T @ X0M, ej)
    tail = X0M @ gram_solution

    eta = np.zeros(n)
    eta[-n0:] = tail
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if Y.size != n:
        raise ValueError(f"Y has size {Y.size}, but the test statistic expects n={n}" )
    etaTY = float(eta @ Y)
    return eta.reshape(-1, 1), etaTY


def _normal_interval_masses(left, right, mean, sigma):
    """Gaussian masses for possibly infinite interval endpoints."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"The Gaussian standard deviation must be positive; got {sigma}" )

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left, right = np.broadcast_arrays(left, right)
    if np.any(np.isnan(left)) or np.any(np.isnan(right)):
        raise ValueError("Gaussian interval endpoints cannot be NaN")

    masses = np.zeros(left.shape, dtype=float)
    valid = right > left
    if not np.any(valid):
        return masses

    z_left = (left[valid] - mean) / sigma
    z_right = (right[valid] - mean) / sigma
    valid_masses = np.empty(z_left.shape, dtype=float)

    positive = z_left >= 0.0
    valid_masses[positive] = (scipy.special.ndtr(-z_left[positive]) - scipy.special.ndtr(-z_right[positive]) )
    valid_masses[~positive] = (scipy.special.ndtr(z_right[~positive]) - scipy.special.ndtr(z_left[~positive]) )
    masses[valid] = np.maximum(valid_masses, 0.0)
    return masses


def calculate_a_b(eta, Y, Sigma):
    eta = np.asarray(eta, dtype=float).reshape(-1, 1)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(Sigma, dtype=float)
    eta_vector = eta[:, 0]
    denominator = float(eta_vector @ Sigma @ eta_vector)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"eta.T @ Sigma @ eta must be positive; got {denominator}" )
    b = Sigma @ eta / denominator
    etaTY = float(eta_vector @ Y[:, 0])
    a = Y - b * etaTY
    return a, b


def calculate_TN_p_value(intervals, eta, etaTY, Sigma, tn_mu=0.0):
    """Paper's p-value P(|Z| >= |etaTY| | Z belongs to the interval union)."""
    intervals = [(left, right) for left, right in merge_intervals(intervals) if right > left ]
    if not intervals:
        raise ValueError("The truncation region is empty")

    variance = float(eta.ravel() @ Sigma @ eta.ravel())
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"eta.T @ Sigma @ eta must be positive; got {variance}")
    sigma = math.sqrt(variance)
    bounds = np.asarray(intervals, dtype=float)
    denominator = float(np.sum(_normal_interval_masses(bounds[:, 0], bounds[:, 1], tn_mu, sigma ) ) )
    if denominator <= 0.0:
        raise ValueError("The truncation region has zero Gaussian mass")

    cutoff = abs(float(etaTY))
    left_tail_mass = np.sum(_normal_interval_masses(bounds[:, 0], np.minimum(bounds[:, 1], -cutoff), tn_mu, sigma) )
    right_tail_mass = np.sum(_normal_interval_masses(np.maximum(bounds[:, 0], cutoff), bounds[:, 1], tn_mu, sigma) )
    numerator = float(left_tail_mass + right_tail_mass)
    return float(np.clip(numerator / denominator, 0.0, 1.0))


__all__ = ["calculate_TN_p_value", "calculate_a_b", "complement_fold_indices", "construct_X_tilde", "construct_Y_tilde", "construct_active_set", "construct_block_slices", "construct_f", "construct_folds", "construct_test_statistic", "construct_w_tilde", "intersect_interval_lists", "merge_intervals", "point_in_interval_list", "solve_quadratic_ineq"]
