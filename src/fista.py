"""This file contains implementation of the LogisticLassoFistaCV class."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from utils import *


class LogisticLassoFistaCV:
    """
    A class for performing logistic regression with Lasso regularization
    using the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) method.
    """

    def __init__(
        self,
        lambdas: ArrayLike = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        warm_start: bool = True,
        report_interval: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the model with specified parameters.

        Args:
            lambdas (ArrayLike, optional): Regularization strengths to be used. Defaults to None.
            max_iter (int, optional): Maximum number of iterations for the optimization algorithm. Defaults to 100.
            tol (float):
            warm_start (bool, optional): Whether to reuse the solution of the previous call to fit as initialization. Defaults to True.
            report_interval (int, optional): Interval for reporting progress during fitting. Defaults to 10.
            verbose (bool, optional): If True, prints detailed information during fitting. Defaults to False.
        """

        self.lambdas = lambdas if lambdas is not None else np.logspace(-4, 1, 80)

        # estimated parameters
        self.coefs_paths_ = {}  # {lambda : beta}
        self.all_validation_scores_ = []  # shape (n_validate_calls, n_lambdas)
        self.best_lambda_ = None
        self.best_beta_ = None
        self.best_lambda_1se_ = None
        self.best_beta_1se_ = None
        self.fitted = False
        self.validated = False

        # training parameters
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.report_interval = report_interval
        self.verbose = verbose

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> "LogisticLassoFistaCV":
        """
        Fit the model according to the given training data.

        Args:
            X_train (ArrayLike): Training matrix of size (n_samples, n_features).
            y_train (ArrayLike): Target vector relative to X.

        Returns:
            LogisticLassoFistaCV: Fitted estimator.
        """
        assert len(X_train.shape) == 2

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Intercept addition
        n, p = X_train.shape[0], X_train.shape[1]

        L = (1 / (4 * n)) * np.linalg.norm(X_train.T @ X_train, ord=2)  # Lipschitz constant

        current_beta = np.zeros(p)

        for reg in sorted(self.lambdas, reverse=True):
            beta = fista(
                X_train,
                y_train,
                current_beta,
                L,
                reg,
                self.max_iter,
                self.tol,
                self.report_interval,
                self.verbose,
            )
            self.coefs_paths_[reg] = beta.copy()

            if self.warm_start:
                current_beta = beta
            else:
                current_beta = np.zeros(p)

        mid_lambda = sorted(self.coefs_paths_.keys())[len(self.coefs_paths_) // 2]
        self.best_beta_ = self.coefs_paths_[mid_lambda]
        self.fitted = True
        return self

    def validate(
        self,
        X_valid: ArrayLike,
        y_valid: ArrayLike,
        measure: str,
        prob_threshold: float = 0.5,
    ) -> tuple[float, float]:
        """
        Validate the fitted model using the validation data.

        Args:
            X_valid (ArrayLike): The validation feature data.
            y_valid (ArrayLike): The true labels for the validation data.
            measure (str):  Measure to be used for validation (e.g., 'balaced_accuracy', 'f1', etc.).
            prob_threshold (float, optional): Threshold for classifying probabilities. Defaults to 0.5.

        Returns:
            tuple[float, float]: A tuple containing the best mean score and the
                                best mean score with one standard error.
        """
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

    def predict_proba(self, X_test: ArrayLike, _1se: bool = False) -> ArrayLike:
        """
        Predict the probability estimates for the given input data.

        Args:
            X_test (ArrayLike): Input data for which to predict probabilities.
            _1se (bool, optional): Whether to use beta that gives the best mean
                score or the best mean score within one standard error. Defaults to False.

        Returns:
            ArrayLike: The predicted probabilities for each class for the input samples.
        """
        assert self.fitted, "Model is not fitted yet. Use 'fit' first."
        # assert self.validated, "Model is not validated yet. Use 'validate' first"
        assert len(X_test.shape) == 2
        #  assert X_test.shape[1] == self.best_beta_.shape[0] - 1

        beta = self.best_beta_1se_ if _1se and self.best_beta_1se_ is not None else self.best_beta_

        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return sigmoid(X_test @ beta)

    def predict(self, X_test: ArrayLike, prob_threshold: float = 0.5, _1se: bool = False) -> ArrayLike:
        """
        Predict the output for the given input data.

        Args:
            X_test (ArrayLike): Input data for which predictions are to be made.
            prob_threshold (float, optional): Threshold for classifying probabilities. Defaults to 0.5.
            _1se (bool, optional): Whether to use beta that gives the best mean
                score or the best mean score within one standard error. Defaults to False.

        Returns:
            ArrayLike: Predicted output based on the input data.
        """
        assert self.fitted, "Model is not fitted yet. Use 'fit' first."
        # assert self.validated, "Model is not validated yet. Use 'validate' first"
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        beta = self.best_beta_1se_ if _1se and self.best_beta_1se_ is not None else self.best_beta_

        y_prob = sigmoid(X_test @ beta)
        y_pred = (y_prob >= prob_threshold).astype(int)
        return y_pred

    def plot_coefficients(self, figsize: tuple[int, int] = (10, 6)) -> None:
        """
        Plot the coefficients of the model against the regularization parameter (lambda).

        Args:
            figsize (tuple[int, int], optional): The size of the figure to be created. Defaults to (10, 6).
        """
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
        """
        Plots the mean validation scores against the regularization parameter (lambda).

        Args:
            measure (str):  Measure to be used for validation (e.g., 'balanced_accuracy', 'f1', etc.).

        """
        assert self.all_validation_scores_, "Model is not validated yet. Use 'validate' first"

        lambdas = sorted(self.lambdas)
        scores_matrix = np.array(self.all_validation_scores_)  # shape: (k_splits, p_lambdas)

        mean_scores = np.mean(scores_matrix, axis=0)
        std_scores = np.std(scores_matrix, axis=0)

        plt.figure(figsize=(10, 6))

        plt.scatter(lambdas, mean_scores, label="Mean Score", color="red", s=25)

        if scores_matrix.shape[0] > 1:
            plt.errorbar(
                lambdas,
                mean_scores,
                std_scores,
                ecolor="gray",
                linestyle="None",
                capsize=2,
            )

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
