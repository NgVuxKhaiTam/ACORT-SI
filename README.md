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

## Package structure

```
ACORT-SI/
├── si_acort/                         # Source code package
│   ├── SI_ACoRT.py                   # Main inference entry points
│   ├── algorithms.py                 # Observed Lasso, source selection, CoRT
│   ├── homotopy.py                   # Critical-cone weighted-Lasso homotopy
│   ├── sub_prob.py                   # Selection and truncation regions
│   ├── utils.py                      # Matrix, interval, and p-value utilities
│   └── gen_data.py                   # Synthetic data generation
├── examples/
│   ├── ex1_p_value_SI_ACoRT.ipynb
│   └── ex2_pivot.ipynb
└── README.md
```

## Usage

```python
import numpy as np
from si_acort import SI_ACoRT, generate_synthetic_data

p = 30
X_list, Y_list, Sigma_list, _ = generate_synthetic_data(
    p=p, s=5, K=3, num_good_sources=2, n0=50, nk=50,
    true_beta=0.5, h=1.0, seed=42,
)
n_list = [X.shape[0] for X in X_list]
lambda_list = [0.6 * np.sqrt(2.0 * np.log(p) / n) for n in n_list]

p_values = SI_ACoRT(
    X_list, Y_list, lambda_list, lam=0.25, Sigma_list=Sigma_list,
    T=5, solver="homotopy",
)
```

`p_values` is `None` if no target feature is selected; otherwise it is a list
of `(feature_index, selective_p_value)` pairs.
