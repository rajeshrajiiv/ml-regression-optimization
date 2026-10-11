import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)


def make_regression_data(n_samples=200, correlation=0.0, noise_std=1.0):
    """Two features whose correlation controls cond(X^T X) -- that condition
    number, not the correlation itself, is what slows gradient descent."""
    z = rng.normal(size=n_samples)
    scale = np.sqrt(max(1 - correlation**2, 1e-6))
    x1 = z + rng.normal(scale=scale, size=n_samples)
    x2 = correlation * z + rng.normal(scale=scale, size=n_samples)
    X = np.column_stack([np.ones(n_samples), x1, x2])
    w_true = np.array([2.0, -3.0, 5.0])
    y = X @ w_true + rng.normal(scale=noise_std, size=n_samples)
    return X, y, w_true


def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def gradient_descent(X, y, lr=0.05, n_iters=500):
    n, d = X.shape
    w = np.zeros(d)
    losses = []
    for _ in range(n_iters):
        residual = X @ w - y
        losses.append((residual @ residual) / (2 * n))
        grad = (X.T @ residual) / n
        w -= lr * grad
    return w, losses


def iters_to_converge(X, y, tol=1e-6, lr=0.05, max_iters=5000):
    n, _ = X.shape
    w = np.zeros(X.shape[1])
    w_star = normal_equation(X, y)
    for i in range(max_iters):
        residual = X @ w - y
        grad = (X.T @ residual) / n
        w -= lr * grad
        if np.linalg.norm(w - w_star) < tol:
            return i + 1
    return max_iters


def main():
    X, y, w_true = make_regression_data(correlation=0.0)
    w_normal = normal_equation(X, y)
    w_gd, losses = gradient_descent(X, y)

    print("True weights:     ", w_true)
    print("Normal equation:  ", w_normal)
    print("Gradient descent: ", w_gd)
    print("Max |w_gd - w_normal|:", np.max(np.abs(w_gd - w_normal)))

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.title("Gradient descent convergence (low feature correlation)")
    plt.savefig(PLOTS_DIR / "gd_convergence.png", dpi=150)

    correlations = [0.0, 0.5, 0.9, 0.95, 0.99]
    conds, iters = [], []
    for c in correlations:
        Xc, yc, _ = make_regression_data(correlation=c)
        cond = np.linalg.cond(Xc.T @ Xc)
        n_iter = iters_to_converge(Xc, yc)
        conds.append(cond)
        iters.append(n_iter)
        print(f"correlation={c:.2f}  cond(X^T X)={cond:9.1f}  iters_to_converge={n_iter}")

    plt.figure()
    plt.plot(conds, iters, marker="o")
    plt.xscale("log")
    plt.xlabel("Condition number of X^T X (log scale)")
    plt.ylabel("Gradient descent iterations to converge")
    plt.title("Ill-conditioning slows gradient descent, not the normal equation")
    plt.savefig(PLOTS_DIR / "conditioning_vs_iters.png", dpi=150)


if __name__ == "__main__":
    main()
