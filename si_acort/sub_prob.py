from __future__ import annotations

from collections import defaultdict

import numpy as np

from .algorithms import CoRT_skglm, solve_lasso_skglm
from .homotopy import PathSegment, compute_solution_path
from .utils import (
    complement_fold_indices,
    construct_X_tilde,
    construct_active_set,
    construct_block_slices,
    construct_f,
    construct_w_tilde,
    intersect_interval_lists,
    merge_intervals,
    point_in_interval_list,
    solve_quadratic_ineq,
)


def _construct_coefficient_state(beta, X):
    beta = np.asarray(beta, dtype=float).reshape(-1)
    active = np.flatnonzero(beta != 0.0)
    inactive = np.flatnonzero(beta == 0.0)
    return {
        "active": active,
        "inactive": inactive,
        "sign": np.sign(beta[active]).reshape(-1, 1),
        "X_active": X[:, active],
        "X_inactive": X[:, inactive],
    }


def _solve_linear_inequalities(coefficients, bounds, tol=1e-14):
    left = -np.inf
    right = np.inf
    for coefficient, bound in zip(coefficients, bounds):
        if abs(coefficient) <= tol:
            if bound < -tol:
                return []
            continue
        value = bound / coefficient
        if coefficient > 0.0:
            right = min(right, value)
        else:
            left = max(left, value)
    if right + tol < left:
        return []
    return [(left, right)]


def _compute_state_interval(beta, X, a, b, penalty_weights):
    state = _construct_coefficient_state(beta, X)
    active = state["active"]
    inactive = state["inactive"]
    sign = state["sign"]
    X_active = state["X_active"]
    X_inactive = state["X_inactive"]
    p = X.shape[1]
    intercept = np.zeros((p, 1))
    slope = np.zeros((p, 1))
    active_coefficients = np.empty(0)
    active_bounds = np.empty(0)
    inactive_coefficients = np.empty(0)
    inactive_bounds = np.empty(0)

    if active.size:
        inverse = np.linalg.pinv(X_active.T @ X_active)
        X_active_plus = inverse @ X_active.T
        selector = np.eye(p)[:, active]
        weighted_sign = penalty_weights[active].reshape(-1, 1) * sign
        intercept = selector @ inverse @ (X_active.T @ a - weighted_sign )
        slope = selector @ inverse @ (X_active.T @ b)
        active_coefficients = (-sign * (X_active_plus @ b) ).ravel()
        active_bounds = (sign * (X_active_plus @ a - inverse @ weighted_sign) ).ravel()

    if inactive.size:
        if active.size:
            projection = (np.eye(X.shape[0]) - X_active @ X_active_plus )
            active_term = (X_inactive.T @ X_active_plus.T @ weighted_sign )
        else:
            projection = np.eye(X.shape[0])
            active_term = np.zeros((inactive.size, 1))
        inactive_map = X_inactive.T @ projection
        response_slope = inactive_map @ b
        response_intercept = inactive_map @ a
        inactive_weights = penalty_weights[inactive].reshape(-1, 1)
        inactive_coefficients = np.concatenate([response_slope.ravel(), -response_slope.ravel()] )
        inactive_bounds = np.concatenate([ (inactive_weights - active_term - response_intercept).ravel(), (inactive_weights + active_term + response_intercept).ravel(), ] )

    coefficients = np.concatenate([active_coefficients, inactive_coefficients] )
    bounds = np.concatenate([active_bounds, inactive_bounds])
    interval = _solve_linear_inequalities(coefficients, bounds)
    return intercept.ravel(), slope.ravel(), interval


def _compute_state_path_skglm(X, a, b, penalty_weights, solve, z_min, z_max, step=1e-5, tol=1e-10):
    X = np.asfortranarray(np.asarray(X, dtype=float))
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    penalty_weights = np.asarray(penalty_weights, dtype=float).reshape(-1)
    segments = []
    z = float(z_min)
    while z < z_max:
        beta = solve((a + b * z).ravel())
        intercept, slope, interval = _compute_state_interval(beta, X, a, b, penalty_weights)
        if not interval:
            z += step
            continue
        left = max(float(interval[0][0]), float(z_min))
        right = min(float(interval[0][1]), float(z_max))
        if left - tol <= z <= right + tol:
            segments.append(PathSegment(left, right, intercept, slope) )
            z = right + step
        else:
            z += step
    return segments


def calculate_validation_loss_coefficients(segment, X_val, a_val, b_val):
    residual_a = np.asarray(a_val).ravel() - X_val @ segment.beta_intercept
    residual_b = np.asarray(b_val).ravel() - X_val @ segment.beta_slope
    n_val = len(residual_a)
    A = 0.5 * float(residual_b @ residual_b) / n_val
    B = float(residual_a @ residual_b) / n_val
    C = 0.5 * float(residual_a @ residual_a) / n_val
    return A, B, C


def compute_fold_selection_region(target_path, combined_path, X_val, a_val, b_val, z_min, z_max, tol=1e-8):
    result = []
    i = j = 0
    while i < len(target_path) and j < len(combined_path):
        target_segment = target_path[i]
        combined_segment = combined_path[j]
        left = max(target_segment.z_left, combined_segment.z_left, z_min)
        right = min(target_segment.z_right, combined_segment.z_right, z_max)

        if left <= right + tol:
            A0, B0, C0 = calculate_validation_loss_coefficients(target_segment, X_val, a_val, b_val )
            Ak, Bk, Ck = calculate_validation_loss_coefficients(combined_segment, X_val, a_val, b_val )
            result.extend(intersect_interval_lists([(left, right)], solve_quadratic_ineq(Ak - A0, Bk - B0, Ck - C0), tol) )

        if target_segment.z_right < combined_segment.z_right - tol:
            i += 1
        else:
            j += 1
    return merge_intervals(result, tol)


def compute_majority_selection_region(fold_regions, threshold, z_min, z_max, tol=1e-8 ):
    endpoints = [z_min, z_max]
    for regions in fold_regions:
        for left, right in regions:
            endpoints.extend([max(left, z_min), min(right, z_max)])
    endpoints = sorted(set(float(value) for value in endpoints if z_min <= value <= z_max) )

    result = []
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        if right <= left + tol:
            continue
        midpoint = 0.5 * (left + right)
        votes = sum(point_in_interval_list(midpoint, regions, tol) for regions in fold_regions )
        if votes >= threshold:
            result.append((left, right))
    return merge_intervals(result, tol)


def compute_source_selection_partitions(X_list, folds, lam, a, b, z_min, z_max, z_obs, tol=1e-8, anchor_cache=None):
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd for majority voting")
    if anchor_cache is None:
        anchor_cache = {}

    n_list = [X.shape[0] for X in X_list]
    slices = construct_block_slices(n_list)
    a0, b0 = a[slices[-1]], b[slices[-1]]
    fold_regions = {}

    for t, fold_indices in enumerate(folds):
        train_indices = complement_fold_indices(n_list[-1], fold_indices)
        X_target_train = X_list[-1][train_indices]
        X_target_val = X_list[-1][fold_indices]
        a_target_train = a0[train_indices]
        b_target_train = b0[train_indices]
        a_target_val = a0[fold_indices]
        b_target_val = b0[fold_indices]

        target_key = ("filter-target", t)
        target_path, target_anchor = compute_solution_path(X_target_train, (a_target_train + b_target_train * z_obs).ravel(), b_target_train.ravel(), np.full(X_target_train.shape[1], X_target_train.shape[0] * lam), z_obs, z_min, z_max, beta_anchor=anchor_cache.get(target_key))
        anchor_cache[target_key] = target_anchor.copy()

        for k in range(len(X_list) - 1):
            X_combined = np.vstack([X_list[k], X_target_train])
            a_combined = np.vstack([a[slices[k]], a_target_train])
            b_combined = np.vstack([b[slices[k]], b_target_train])
            combined_key = ("filter-combined", t, k)
            combined_path, combined_anchor = compute_solution_path(X_combined, (a_combined + b_combined * z_obs).ravel(), b_combined.ravel(), np.full(X_combined.shape[1], X_combined.shape[0] * lam), z_obs, z_min, z_max, beta_anchor=anchor_cache.get(combined_key))
            anchor_cache[combined_key] = combined_anchor.copy()
            fold_regions[(t, k)] = compute_fold_selection_region(target_path, combined_path, X_target_val, a_target_val, b_target_val, z_min, z_max, tol)

    threshold = (len(folds) + 1) // 2
    majority_regions = {k: compute_majority_selection_region([fold_regions[(t, k)] for t in range(len(folds))], threshold, z_min, z_max, tol) for k in range(len(X_list) - 1) }

    endpoints = [z_min, z_max]
    for regions in majority_regions.values():
        for left, right in regions:
            endpoints.extend([left, right])
    endpoints = sorted(set(float(value) for value in endpoints))

    partitions = []
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        if right <= left + tol:
            continue
        midpoint = 0.5 * (left + right)
        I = [k for k, regions in majority_regions.items() if point_in_interval_list(midpoint, regions, tol) ]
        if partitions and partitions[-1][2] == I and left <= partitions[-1][1] + tol:
            partitions[-1] = (partitions[-1][0], right, I)
        else:
            partitions.append((left, right, I))
    return partitions


def compute_source_selection_partitions_skglm(X_list, folds, lam, a, b, z_min, z_max, tol=1e-8):
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd for majority voting")

    n_list = [X.shape[0] for X in X_list]
    slices = construct_block_slices(n_list)
    a0 = a[slices[-1]]
    b0 = b[slices[-1]]
    fold_regions = {}

    for t, fold_indices in enumerate(folds):
        train_indices = complement_fold_indices(n_list[-1], fold_indices)
        X_target_train = X_list[-1][train_indices]
        X_target_val = X_list[-1][fold_indices]
        a_target_train = a0[train_indices]
        b_target_train = b0[train_indices]
        a_target_val = a0[fold_indices]
        b_target_val = b0[fold_indices]
        target_weights = np.full(X_target_train.shape[1], X_target_train.shape[0] * lam)
        target_path = _compute_state_path_skglm(X_target_train, a_target_train, b_target_train, target_weights, lambda y: solve_lasso_skglm(X_target_train, y, lam), z_min, z_max)

        for k in range(len(X_list) - 1):
            X_combined = np.vstack([X_list[k], X_target_train])
            a_combined = np.vstack([a[slices[k]], a_target_train])
            b_combined = np.vstack([b[slices[k]], b_target_train])
            combined_weights = np.full(X_combined.shape[1], X_combined.shape[0] * lam)
            combined_path = _compute_state_path_skglm(X_combined, a_combined, b_combined, combined_weights, lambda y, X=X_combined: solve_lasso_skglm(X, y, lam), z_min, z_max)
            fold_regions[(t, k)] = compute_fold_selection_region(target_path, combined_path, X_target_val, a_target_val, b_target_val, z_min, z_max, tol)

    threshold = (len(folds) + 1) // 2
    majority_regions = {k: compute_majority_selection_region([fold_regions[(t, k)] for t in range(len(folds))], threshold, z_min, z_max, tol) for k in range(len(X_list) - 1) }
    endpoints = [z_min, z_max]
    for regions in majority_regions.values():
        for left, right in regions:
            endpoints.extend([left, right])
    endpoints = sorted(set(float(value) for value in endpoints))

    partitions = []
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        if right <= left + tol:
            continue
        midpoint = 0.5 * (left + right)
        I = [k for k, regions in majority_regions.items() if point_in_interval_list(midpoint, regions, tol) ]
        if (
            partitions
            and partitions[-1][2] == I
            and left <= partitions[-1][1] + tol
        ):
            partitions[-1] = (partitions[-1][0], right, I)
        else:
            partitions.append((left, right, I))
    return partitions


def compute_fixed_I_support_region(cort_path, target_start, p, M_obs, z_min, z_max, tol=1e-8):
    observed_support = tuple(M_obs)
    result = []
    for segment in cort_path:
        left = max(segment.z_left, z_min)
        right = min(segment.z_right, z_max)
        if left > right + tol:
            continue
        midpoint = 0.5 * (left + right)
        theta_target = segment.evaluate(midpoint)[target_start : target_start + p]
        if tuple(construct_active_set(theta_target)) == observed_support:
            result.append((left, right))
    return merge_intervals(result, tol)


def compute_Z_homotopy(X_list, folds, lam, a, b, M_obs, lambda_list, n_list, z_min, z_max, z_obs, tol=1e-8, anchor_cache=None):
    if anchor_cache is None:
        anchor_cache = {}
    partitions = compute_source_selection_partitions(X_list, folds, lam, a, b, z_min, z_max, z_obs, tol, anchor_cache)

    regions_by_I = defaultdict(list)
    for left, right, I in partitions:
        regions_by_I[tuple(I)].append((left, right))

    slices = construct_block_slices(n_list)
    p = X_list[0].shape[1]
    Z = []
    for I_tuple, source_regions in regions_by_I.items():
        I = list(I_tuple)
        X_tilde = construct_X_tilde(X_list, I)
        f_tilde = construct_f(n_list, I)
        w_tilde = construct_w_tilde(p, lambda_list, I)
        a_I = np.vstack([a[slices[k]] for k in I + [-1]])
        b_I = np.vstack([b[slices[k]] for k in I + [-1]])

        cort_key = ("cort", I_tuple)
        cort_path, cort_anchor = compute_solution_path(X_tilde, (f_tilde * (a_I + b_I * z_obs)).ravel(), (f_tilde * b_I).ravel(), w_tilde, z_obs, z_min, z_max, beta_anchor=anchor_cache.get(cort_key))
        anchor_cache[cort_key] = cort_anchor.copy()
        support_region = compute_fixed_I_support_region(cort_path, len(I) * p, p, M_obs, z_min, z_max, tol)
        Z.extend(intersect_interval_lists(source_regions, support_region, tol))

    return merge_intervals(Z, tol)


def compute_Z_skglm(X_list, folds, lam, a, b, M_obs, lambda_list, n_list, z_min, z_max, tol=1e-8):
    partitions = compute_source_selection_partitions_skglm(X_list, folds, lam, a, b, z_min, z_max, tol)
    regions_by_I = defaultdict(list)
    for left, right, I in partitions:
        regions_by_I[tuple(I)].append((left, right))

    slices = construct_block_slices(n_list)
    p = X_list[0].shape[1]
    Z = []
    for I_tuple, source_regions in regions_by_I.items():
        I = list(I_tuple)
        X_tilde = construct_X_tilde(X_list, I)
        f_tilde = construct_f(n_list, I)
        w_tilde = construct_w_tilde(p, lambda_list, I)
        a_I = np.vstack([a[slices[k]] for k in I + [-1]])
        b_I = np.vstack([b[slices[k]] for k in I + [-1]])
        a_tilde = f_tilde * a_I
        b_tilde = f_tilde * b_I
        cort_path = _compute_state_path_skglm(X_tilde, a_tilde, b_tilde, w_tilde, lambda y, X=X_tilde, w=w_tilde: CoRT_skglm(X, y, w, p)[0], z_min, z_max)
        support_region = compute_fixed_I_support_region(cort_path, len(I) * p, p, M_obs, z_min, z_max, tol)
        Z.extend(intersect_interval_lists(source_regions, support_region, tol) )
    return merge_intervals(Z, tol)


def compute_Z(X_list, folds, lam, a, b, M_obs, lambda_list, n_list, z_min, z_max, z_obs, tol=1e-8, anchor_cache=None, solver="homotopy"):
    if solver == "homotopy":
        return compute_Z_homotopy(X_list, folds, lam, a, b, M_obs, lambda_list, n_list, z_min, z_max, z_obs, tol, anchor_cache)
    if solver == "skglm":
        return compute_Z_skglm(X_list, folds, lam, a, b, M_obs, lambda_list, n_list, z_min, z_max, tol)
    raise ValueError("solver must be 'homotopy' or 'skglm'")


__all__ = ["calculate_validation_loss_coefficients", "compute_Z", "compute_Z_homotopy", "compute_Z_skglm", "compute_fixed_I_support_region", "compute_fold_selection_region", "compute_majority_selection_region", "compute_source_selection_partitions", "compute_source_selection_partitions_skglm"]
