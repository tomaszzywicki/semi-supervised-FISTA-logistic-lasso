"""This file contains helper functions."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def fista(X, y, beta, L, reg, max_iter, tol, report_interval: int, verbose):
    t = 1
    c = beta

    patience = 3
    tol_counter = 0

    for k in range(max_iter):
        z = c + (1 / L) * gradient(X, y, c)  # Gradient step
        beta_new = soft_threshold_l1(z, L, reg)  # soft thresholding
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2  # t update
        c = beta_new + ((t - 1) / t_new) * (beta_new - beta)  # Better step selection

        diff = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-8)
        if diff < tol:
            tol_counter += 1
            if tol_counter >= patience:
                if verbose:
                    print(f"FISTA converged at iter {k} | diff: {diff:.6e} ")
                break
        else:
            tol_counter = 0

        beta = beta_new
        t = t_new

        log_lik = -log_likelihood_l1(X, y, beta, reg)
        if verbose and k % report_interval == 0:
            print(
                f"Iter [{k:3d}/{max_iter}]. Difference: {diff:.9f} | "
                # f"Beta: {np.array2string(beta, formatter={"float_kind": lambda x: f"{x:.5f}"})}"
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
    z = np.clip(z, -500, 500)
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


def calculate_metric_value(
    y_true: np.array, y_pred: np.array, y_prob: np.array, measure: str
) -> float:
    metrics_funs = {
        "recall": lambda: recall_score(y_true, y_pred),
        "precision": lambda: precision_score(y_true, y_pred),
        "f1": lambda: f1_score(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred),
        "roc": lambda: roc_auc_score(y_true, y_prob),
        "pr_auc": lambda: average_precision_score(y_true, y_prob),
    }

    if measure not in metrics_funs:
        raise ValueError(
            f"Error: Invalid measure '{measure}'. Select from: {list(metrics_funs.keys())}"
        )

    return metrics_funs[measure]()


def load_arff(filepath):
    columns = []
    data = []
    in_data = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("%") or not line:
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                parts = line.split(None, 2)
                col_name = parts[1]
                columns.append(col_name)
            elif line.upper().startswith("@DATA"):
                in_data = True
            elif in_data:
                data.append(line.split(","))

    return pd.DataFrame(data, columns=columns)


def set_best_st_approach(df):
    df_st = df[df["Approach"] == "self_training"].copy()
    df_st["Config"] = df_st["base_estimator"] + "_k=" + df_st["k_best"].astype(str)

    # best in terms of median for F1
    global_medians = df_st.groupby("Config")["F1"].median()
    best_config = global_medians.idxmax()
    print(f"Best Self-Training configuration: {best_config}")

    best_estimator = best_config.split("_k=")[0]
    best_k =int(float(best_config.split('_k=')[1]))
    # best_k = 1
    mask_other = df["Approach"] != "self_training"
    mask_best_st = (
        (df["Approach"] == "self_training")
        & (df["base_estimator"] == best_estimator)
        & (df["k_best"] == int(best_k))
    )

    df_plot = df[mask_other | mask_best_st].copy()

    df_plot.loc[df_plot["Approach"] == "self_training", "Approach"] = (
        f"Self-Training ({best_config})"
    )
    return df_plot
