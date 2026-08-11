"""Synthetic-data generator used by the SI-ACoRT examples."""

from __future__ import annotations

import numpy as np


def generate_synthetic_data(p=500, s=10, K=10, num_good_sources=7, n0=50, nk=100, true_beta=0.25, h=5, rho=0.5, source_shift_sd=0.3, sigma_noise=1.0, seed=None):
    beta_0 = np.concatenate([np.full(s, true_beta), np.zeros(p - s)])
    indices = np.arange(p)
    Sigma_X0 = rho ** np.abs(indices[:, None] - indices[None, :])
    random = np.random if seed is None else np.random.RandomState(seed)

    X0 = random.multivariate_normal(np.zeros(p), Sigma_X0, size=n0)
    Y0 = X0 @ beta_0 + random.normal(0.0, sigma_noise, n0)

    X_list = []
    Y_list = []
    Sigma_list = []
    for k in range(K):
        signs = random.choice([-1.0, 1.0], size=p)
        if k < num_good_sources:
            beta_k = beta_0 + (h / p) * signs
        else:
            beta_k = 2.0 * h * signs
            fixed_poison_idx = np.arange(s, 2 * s)
            random_poison_idx = random.choice(np.arange(2 * s, p), size=s, replace=False )
            poison_idx = np.concatenate([fixed_poison_idx, random_poison_idx] )
            beta_k[poison_idx] = (0.5 + (2.0 * h * signs[poison_idx]) / p )

        shift = random.normal(0.0, source_shift_sd, size=(p, 1))
        Sigma_Xk = Sigma_X0 + shift @ shift.T
        Xk = random.multivariate_normal(np.zeros(p), Sigma_Xk, size=nk)
        Yk = Xk @ beta_k + random.normal(0.0, sigma_noise, nk)
        X_list.append(Xk)
        Y_list.append(Yk)
        Sigma_list.append(sigma_noise**2 * np.eye(nk))

    X_list.append(X0)
    Y_list.append(Y0)
    Sigma_list.append(sigma_noise**2 * np.eye(n0))
    return X_list, Y_list, Sigma_list, beta_0


__all__ = ["generate_synthetic_data"]
