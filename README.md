# ML Math Foundations: Regression & Convex Optimization

## What this actually shows

1. `linear_regression.py` — the Normal Equation computes the least-squares solution directly, while gradient descent reaches the same answer through repeated updates. As features become more correlated and cond(XᵀX) increases, gradient descent needs significantly more iterations. The results show how the shape of the loss surface affects convergence in practice.

2. `logistic_regression.py` — We computed the Hessian of the cross-entropy loss and checked that it was positive semidefinite at several weight vectors. These checks are consistent with the loss being convex, meaning gradient descent has no non-global local minima to get trapped in.

## Run it

```bash
pip install -r requirements.txt
python linear_regression.py
python logistic_regression.py
```

Plots are written to `plots/`.

## Output

- `plots/gd_convergence.png` — MSE loss vs. iteration for gradient descent
  on a well-conditioned regression problem.
- `plots/conditioning_vs_iters.png` — iterations-to-converge rises sharply
  as feature correlation (and `cond(XᵀX)`) increases, while the Normal
  Equation's answer is unaffected.
- `plots/logreg_convergence.png` — cross-entropy loss vs. iteration.
- `plots/logreg_boundary.png` — learned decision boundary on synthetic
  two-class data.

