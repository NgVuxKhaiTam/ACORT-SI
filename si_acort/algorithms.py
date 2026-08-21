from __future__ import annotations

import numpy as np

from .homotopy import solve_weighted_lasso
from .utils import complement_fold_indices


def solve_lasso_homotopy(X, Y, lam):
    X = np.asfortranarray(np.asarray(X, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(-1)
    penalty_weights = np.full(X.shape[1], X.shape[0] * lam)
    return solve_weighted_lasso(X, Y, penalty_weights)


def solve_lasso_skglm(X, Y, lam):
    from skglm import Lasso

    X = np.asfortranarray(np.asarray(X, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(-1)
    model = Lasso(alpha=lam, fit_intercept=False, tol=1e-11)
    model.fit(X, Y)
    return model.coef_


def solve_lasso(X, Y, lam, solver="homotopy"):
    if lam <= 0.0:
        raise ValueError("lam must be strictly positive")
    if solver == "homotopy":
        return solve_lasso_homotopy(X, Y, lam)
    if solver == "skglm":
        return solve_lasso_skglm(X, Y, lam)
    raise ValueError("solver must be 'homotopy' or 'skglm'")


def CoRT_homotopy(X_tilde, Y_tilde, w_tilde, p):
    X_tilde = np.asfortranarray(np.asarray(X_tilde, dtype=float))
    Y_tilde = np.asarray(Y_tilde, dtype=float).reshape(-1)
    w_tilde = np.asarray(w_tilde, dtype=float).reshape(-1)
    theta_hat = solve_weighted_lasso(X_tilde, Y_tilde, w_tilde)
    return theta_hat, theta_hat[-p:]


def CoRT_skglm(X_tilde, Y_tilde, w_tilde, p):
    from skglm import WeightedLasso

    X_tilde = np.asfortranarray(np.asarray(X_tilde, dtype=float))
    Y_tilde = np.asarray(Y_tilde, dtype=float).reshape(-1)
    w_tilde = np.asarray(w_tilde, dtype=float).reshape(-1)
    model = WeightedLasso(alpha=1.0 / Y_tilde.size, fit_intercept=False, tol=1e-13, weights=w_tilde)
    model.fit(X_tilde, Y_tilde)
    theta_hat = model.coef_
    return theta_hat, theta_hat[-p:]


def CoRT(X_tilde, Y_tilde, w_tilde, p, solver="homotopy"):
    if p <= 0 or p > np.asarray(X_tilde).shape[1]:
        raise ValueError("p is incompatible with the CoRT coefficient dimension")
    if solver == "homotopy":
        return CoRT_homotopy(X_tilde, Y_tilde, w_tilde, p)
    if solver == "skglm":
        return CoRT_skglm(X_tilde, Y_tilde, w_tilde, p)
    raise ValueError("solver must be 'homotopy' or 'skglm'")


def adaptive_source_selection(X_list, Y_list, folds, lam, solver="homotopy"):
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd for majority voting")
    if len(X_list) != len(Y_list) or len(X_list) < 1:
        raise ValueError("X_list and Y_list must contain the same data blocks")

    X0 = np.asarray(X_list[-1], dtype=float)
    Y0 = np.asarray(Y_list[-1], dtype=float).reshape(-1)
    votes = np.zeros(len(X_list) - 1, dtype=int)

    for fold_indices in folds:
        train_indices = complement_fold_indices(len(Y0), fold_indices)
        X0_train = X0[train_indices]
        Y0_train = Y0[train_indices]
        X0_val = X0[fold_indices]
        Y0_val = Y0[fold_indices]
        beta0 = solve_lasso(X0_train, Y0_train, lam, solver)
        loss0 = 0.5 * np.mean((Y0_val - X0_val @ beta0) ** 2)

        for k in range(len(X_list) - 1):
            X_combined = np.vstack([X_list[k], X0_train])
            Y_combined = np.concatenate([Y_list[k], Y0_train])
            betak = solve_lasso(X_combined, Y_combined, lam, solver)
            lossk = 0.5 * np.mean((Y0_val - X0_val @ betak) ** 2)
            if lossk <= loss0:
                votes[k] += 1

    threshold = (len(folds) + 1) // 2
    return np.flatnonzero(votes >= threshold).tolist()


__all__ = ["CoRT", "CoRT_homotopy", "CoRT_skglm", "adaptive_source_selection", "solve_lasso", "solve_lasso_homotopy", "solve_lasso_skglm"]
