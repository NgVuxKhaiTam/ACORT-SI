import math

import numpy as np
import scipy.linalg

from .algorithms import CoRT, adaptive_source_selection
from .sub_prob import compute_Z
from .utils import (
    calculate_TN_p_value,
    calculate_a_b,
    construct_X_tilde,
    construct_Y_tilde,
    construct_active_set,
    construct_folds,
    construct_test_statistic,
    construct_w_tilde,
)


def construct_observed_state(X_list, Y_list, lambda_list, lam, T=5, solver="homotopy"):
    if solver not in {"homotopy", "skglm"}:
        raise ValueError("solver must be 'homotopy' or 'skglm'")
    if lam <= 0.0 or any(value <= 0.0 for value in lambda_list):
        raise ValueError("The penalty parameters must be strictly positive")
    if not 0 < T <= len(Y_list[-1]) or T % 2 == 0:
        raise ValueError("T must be odd and between 1 and the target sample size")
    folds = construct_folds(len(Y_list[-1]), T)

    X_list = [np.asarray(X, dtype=float) for X in X_list]
    Y_list = [np.asarray(Y, dtype=float).reshape(-1) for Y in Y_list]
    p = X_list[0].shape[1]

    n_list = [X.shape[0] for X in X_list]
    I_obs = adaptive_source_selection(X_list, Y_list, folds, lam, solver)
    X_tilde = construct_X_tilde(X_list, I_obs)
    Y_tilde = construct_Y_tilde(Y_list, n_list, I_obs)
    w_tilde = construct_w_tilde(p, lambda_list, I_obs)
    theta_hat, beta0_hat = CoRT(X_tilde, Y_tilde, w_tilde, p, solver)
    M_obs = construct_active_set(beta0_hat)
    return {
        "X_list": X_list,
        "Y_list": Y_list,
        "folds": folds,
        "n_list": n_list,
        "I_obs": I_obs,
        "theta_hat": theta_hat,
        "beta0_hat": beta0_hat,
        "M_obs": M_obs,
        "solver": solver,
    }


def calculate_feature_p_value(observed_state, j, lambda_list, lam, Sigma_list, threshold=20.0, anchor_cache=None):
    X_list = observed_state["X_list"]
    Y_list = observed_state["Y_list"]
    folds = observed_state["folds"]
    n_list = observed_state["n_list"]
    M_obs = observed_state["M_obs"]
    n0 = n_list[-1]
    n = sum(n_list)
    Y = np.concatenate(Y_list)
    Sigma = scipy.linalg.block_diag(*Sigma_list)
    X0M = X_list[-1][:, M_obs]
    eta, etaTY = construct_test_statistic(j, X0M, Y, M_obs, n0, n)
    stdev = math.sqrt(float(eta.ravel() @ Sigma @ eta.ravel()))
    z_obs = float(etaTY)
    z_min = min(-threshold * stdev, z_obs)
    z_max = max(threshold * stdev, z_obs)
    a, b = calculate_a_b(eta, Y, Sigma)

    Z = compute_Z(X_list, folds, lam, a, b, M_obs, lambda_list, n_list, z_min, z_max, z_obs, anchor_cache=anchor_cache, solver=observed_state["solver"])
    p_value = calculate_TN_p_value(Z, eta, etaTY, Sigma)
    return p_value


def SI_ACoRT(X_list, Y_list, lambda_list, lam, Sigma_list, T=5, threshold=20.0, solver="homotopy"):
    observed_state = construct_observed_state(X_list, Y_list, lambda_list, lam, T, solver)
    if not observed_state["M_obs"]:
        return None

    anchor_cache = {}
    return [(j, calculate_feature_p_value(observed_state, j, lambda_list, lam, Sigma_list, threshold, anchor_cache)) for j in observed_state["M_obs"]]
