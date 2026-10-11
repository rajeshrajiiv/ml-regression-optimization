import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(7)


def make_classification_data(n_per_class=150):
    mean0 = np.array([-1.5, -1.0])
    mean1 = np.array([1.5, 1.5])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.column_stack([np.ones(2 * n_per_class), np.vstack([X0, X1])])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy_loss(X, y, w):
    p = sigmoid(X @ w)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def gradient(X, y, w):
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)


def hessian(X, w):
    """d^2/dw^2 of the cross-entropy loss. PSD everywhere <=> loss is convex."""
    p = sigmoid(X @ w)
    S = p * (1 - p)
    return (X.T * S) @ X / X.shape[0]


def gradient_descent(X, y, lr=0.5, n_iters=300):
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(n_iters):
        losses.append(cross_entropy_loss(X, y, w))
        w -= lr * gradient(X, y, w)
    return w, losses


def check_convexity(X, w_samples):
    """Verify convexity isn't just a claim from the lecture notes: the Hessian
    of cross-entropy loss must be positive semi-definite at every weight vector."""
    for w in w_samples:
        eigvals = np.linalg.eigvalsh(hessian(X, w))
        print(f"w={np.round(w, 2)}  min eigenvalue of Hessian={eigvals.min():.6f}")
        assert eigvals.min() >= -1e-8, "Hessian not PSD -- loss would not be convex here"


def main():
    X, y = make_classification_data()
    w, losses = gradient_descent(X, y)
    preds = (sigmoid(X @ w) >= 0.5).astype(float)
    print("Learned weights:", w)
    print(f"Training accuracy: {np.mean(preds == y):.3f}")

    print("\nConvexity check (Hessian must be PSD at every point evaluated):")
    w_samples = [np.zeros(3), w, rng.normal(size=3), rng.normal(size=3) * 5]
    check_convexity(X, w_samples)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("Logistic regression convergence")
    plt.savefig(PLOTS_DIR / "logreg_convergence.png", dpi=150)

    plt.figure()
    plt.scatter(X[y == 0, 1], X[y == 0, 2], label="class 0", alpha=0.6)
    plt.scatter(X[y == 1, 1], X[y == 1, 2], label="class 1", alpha=0.6)
    x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2_boundary = -(w[0] + w[1] * x1_range) / w[2]
    plt.plot(x1_range, x2_boundary, "k--", label="decision boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Logistic regression decision boundary")
    plt.savefig(PLOTS_DIR / "logreg_boundary.png", dpi=150)


if __name__ == "__main__":
    main()
