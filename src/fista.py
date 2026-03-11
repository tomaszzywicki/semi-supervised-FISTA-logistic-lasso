from re import A
import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

from numpy.typing import NDArray, ArrayLike


class LogisticLassoFistaCV:

    def __init__(self, lambdas: ArrayLike = None):
        self.lambdas = lambdas if lambdas is not None else np.logspace(-4, 1, 80)

        self.coefs_paths_ = {}  # {lambda : beta}
        self.all_validation_scores_ = []  # shape (n_validate_calls, n_lambdas)
        self.best_lambda_ = None
        self.best_beta_ = None
        self.best_lambda_1se_ = None
        self.best_beta_1se_ = None
        self.fitted = False
        self.validated = False

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        max_iter: int = 100,
        warm_start: bool = True,
        report_interval: int = 10,
        verbose: bool = False,
    ) -> None:
        # TODO dodać żeby patrzyło czy zbiegło przed max_iter
        assert len(X_train.shape) == 2

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Intercept addition
        n, p = X_train.shape[0], X_train.shape[1]

        L = (1 / (4 * n)) * np.linalg.norm(X_train.T @ X_train, ord=2)  # Lipschitz constant

        current_beta = np.zeros(p)

        for reg in sorted(self.lambdas, reverse=True):
            beta = fista(X_train, y_train, current_beta, L, reg, max_iter, report_interval, verbose)
            self.coefs_paths_[reg] = beta.copy()

            if warm_start:
                current_beta = beta
            else:
                current_beta = np.zeros(p)

        self.fitted = True
        return

    def validate(self, X_valid: ArrayLike, y_valid: ArrayLike, measure: str, prob_threshold: float = 0.5):
        assert self.fitted, "Model is not fitted yet. Use 'fit' first."
        X_valid = np.hstack((np.ones((X_valid.shape[0], 1)), X_valid))

        currect_scores = []
        lambdas_sorted = sorted(self.lambdas)

        for reg in lambdas_sorted:
            beta = self.coefs_paths_[reg]
            y_prob = sigmoid(X_valid @ beta)
            y_pred = (y_prob >= prob_threshold).astype(int)

            score = calculate_metric_value(y_valid, y_pred, y_prob, measure)
            currect_scores.append(score)

        self.all_validation_scores_.append(currect_scores)
        scores_matrix = np.asarray(self.all_validation_scores_)
        k_splits = scores_matrix.shape[0]

        mean_scores = np.mean(scores_matrix, axis=0)
        std_scores = np.std(scores_matrix, axis=0)

        best_idx = np.argmax(mean_scores)
        best_score = mean_scores[best_idx]
        best_idx_1_se = best_idx

        se = std_scores[best_idx] / np.sqrt(k_splits) if k_splits > 1 else 0
        threshold = best_score - se

        for i in range(len(lambdas_sorted) - 1, -1, -1):
            if mean_scores[i] >= threshold:
                best_idx_1_se = i
                break

        self.best_lambda_1se_ = lambdas_sorted[best_idx_1_se]
        self.best_beta_1se_ = self.coefs_paths_[self.best_lambda_1se_]

        self.best_lambda_ = lambdas_sorted[best_idx]
        self.best_beta_ = self.coefs_paths_[self.best_lambda_]

        return mean_scores[best_idx], mean_scores[best_idx_1_se]

    def predict_proba(self, X_test: ArrayLike, _1se: bool = False):
        assert self.fitted, "Model is not fitted yet. Use 'fit' first."
        assert self.validated, "Model is not validated yet. Use 'validate' first"
        assert len(X_test.shape) == 2
        assert X_test.shape[1] == self.best_beta_.shape[0] - 1

        beta = self.best_beta_1se_ if _1se else self.best_beta_

        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return sigmoid(X_test @ beta)

    def predict(self, X_test: ArrayLike, prob_threshold: float = 0.5, _1se: bool = False):
        assert self.fitted, "Model is not fitted yet. Use 'fit' first."
        assert self.validated, "Model is not validated yet. Use 'validate' first"

        beta = self.best_beta_1se_ if _1se else self.best_beta_

        y_prob = sigmoid(X_test @ beta)
        y_pred = (y_prob >= prob_threshold).astype(int)
        return y_pred

    def plot_coefficients(self, figsize: tuple[int, int] = (10, 6)) -> None:
        assert self.fitted, "Model is not fitted yet. Use 'fit' first."

        lambdas_sorted = sorted(self.lambdas)
        coefs = [self.coefs_paths_[l][1:] for l in lambdas_sorted]

        plt.figure(figsize=figsize)
        plt.plot(lambdas_sorted, coefs, alpha=0.8)
        plt.xscale("log")
        plt.xlabel(r"$\lambda$")
        plt.ylabel("Coefficients")
        plt.title("Lasso Regularization Path")
        plt.axis("tight")

    def plot(self, measure: str) -> None:
        assert self.all_validation_scores_, "Model is not validated yet. Use 'validate' first"

        lambdas = sorted(self.lambdas)
        scores_matrix = np.array(self.all_validation_scores_)  # shape: (k_splits, p_lambdas)

        mean_scores = np.mean(scores_matrix, axis=0)
        std_scores = np.std(scores_matrix, axis=0)

        plt.figure(figsize=(10, 6))

        plt.scatter(lambdas, mean_scores, label="Mean Score", color="red", s=25)

        if scores_matrix.shape[0] > 1:
            plt.errorbar(lambdas, mean_scores, std_scores, ecolor="gray", linestyle="None", capsize=2)

        plt.axvline(
            self.best_lambda_,
            linestyle="--",
            color="grey",
            label=rf"Best $\lambda$: {self.best_lambda_:.4f}",
            alpha=0.7,
        )
        if self.best_lambda_1se_ != self.best_lambda_:
            plt.axvline(
                self.best_lambda_1se_,
                linestyle="--",
                color="lightgrey",
                label=r"Best $\lambda_{1se}$:" + f"{self.best_lambda_1se_:.4f}",
                alpha=1,
            )
        plt.xscale("log")
        plt.xlabel("Lambda")
        plt.ylabel(f"{measure} measure")
        plt.title(f"{measure} measure vs Lambda (Repeated Holdout, n_splits={scores_matrix.shape[0]})")
        plt.legend()
        plt.show()


def fista(X, y, beta, L, reg, max_iter, report_interval: int, verbose):
    t = 1
    c = beta

    for k in range(max_iter):
        z = c + (1 / L) * gradient(X, y, c)  # Gradient step
        beta_new = soft_threshold_l1(z, L, reg)  # soft thresholding
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2  # t update
        c = beta_new + ((t - 1) / t_new) * (beta_new - beta)  # Better step selection

        beta = beta_new
        t = t_new

        log_lik = -log_likelihood_l1(X, y, beta, reg)
        if verbose and k % report_interval == 0:
            print(
                f"Iter [{k:3d}/{max_iter}]. Log-lik: {log_lik:.4f} |"
                f" Beta: {np.array2string(beta, formatter={'float_kind': lambda x: f"{x:.5f}"})}"
            )

    return beta


def log_likelihood_l1(X, y, beta, reg):
    z = X @ beta
    log_1_exp_z = np.maximum(0, z) + np.log(1 + np.exp(-np.abs(z)))
    loss = -np.mean(y * z - log_1_exp_z)
    return loss + reg * np.sum(np.abs(beta[1:]))


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable sigmoid function

    Args:
        z (ArrayLike): Input value(s). Accepts a Python float/int, a list of numbers, or NumPy array of any shape.

    Returns:
        NDArray: Array of sigmoid values, same shape as input
    """
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def gradient(
    X: NDArray[np.float64], y: NDArray[np.float64], beta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Computes the gradient of the log-likelihood for logistic regression.

    Args:
        X (NDArray[np.float64]): Feature matrix of shape (n_samples, n_features).
        y (NDArray[np.float64]): Binary target vector with values 0 or 1 and shape (n_samples,).
        beta (NDArray[np.float64]):  Coefficient vector of shape (n_features,).

    Returns:
        NDArray[np.float64]: Gradient vector of shape (n_features,).
    """
    n = X.shape[0]
    z = X @ beta
    p = sigmoid(z)
    return (X.T @ (y - p)) / n


def soft_threshold_l1(z, L, reg):
    threshold = reg / L
    res = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
    res[0] = z[0]
    return res


def calculate_metric_value(y_true: np.array, y_pred: np.array, y_prob: np.array, measure: str) -> float:
    metrics_funs = {
        "recall": lambda: recall_score(y_true, y_pred),
        "precision": lambda: precision_score(y_true, y_pred),
        "f": lambda: f1_score(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred),
        "roc": lambda: roc_auc_score(y_true, y_prob),
        "pr_auc": lambda: average_precision_score(y_true, y_prob),
    }

    if measure not in metrics_funs:
        raise ValueError(f"Error: Invalid measure '{measure}'. Select from: {list(metrics_funs.keys())}")

    return metrics_funs[measure]()
