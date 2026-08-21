# SI-ACoRT: Selective Inference for Adaptive CoRT

**SI-ACoRT** provides selective p-values after Adaptive CoRT source filtration
and target-feature selection.

## Core properties

- **Less-over-conditioned inference:** unions all source sets compatible with
  the observed target support `M_obs`.
- **Independent solvers:** supports `homotopy` and `skglm` without fallback
  between them. The default solver is `homotopy`.

## Requirements

The `homotopy` solver requires Python 3.8+ with
[`numpy`](https://numpy.org/doc/stable/) and
[`scipy`](https://docs.scipy.org/doc/). The `skglm` solver requires Python 3.9+
and [`skglm>=0.5`](https://contrib.scikit-learn.org/skglm/). The parallel pivot
example also uses [`joblib`](https://joblib.readthedocs.io/).

## Usage

```python
import numpy as np
from si_acort import SI_ACoRT, generate_synthetic_data

p = 500
s = 10
K = 10
n0 = 100
nk = 100

X_list, Y_list, Sigma_list, true_beta0 = generate_synthetic_data(
    p=p, s=s, K=K, num_good_sources=7, n0=n0, nk=nk,
    true_beta=0.5, h=20.0, seed=42,
)
n_list = [X.shape[0] for X in X_list]
lambda_list = [2.0 * np.sqrt(np.log(p) / n) for n in n_list]
lam = 1.0

p_values = SI_ACoRT(
    X_list=X_list, Y_list=Y_list, lambda_list=lambda_list,
    lam=lam, Sigma_list=Sigma_list, T=5,
)

if p_values is None:
    print('No target feature was selected.')
else:
    for j, p_value in p_values:
        print(f'feature={j:3d}  beta*={true_beta0[j]: .3f}  p={p_value:.6f}')
```

`p_values` is `None` if no target feature is selected; otherwise it is a list
of `(feature_index, selective_p_value)` pairs.
