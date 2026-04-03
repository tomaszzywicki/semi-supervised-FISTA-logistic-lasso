"""This file contains implementation of the UnlabeledLogReg class."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import kneighbors_graph

from fista import LogisticLassoFistaCV


class UnlabeledLogReg:
    """
    A class for semi-supervised logistic regression.
    """

    def __init__(
        self,
        y_imputation_method: (
            Literal[
                "random",
                "naive",
                "pseudo_labels",
                "self_training",
                "label_propagation",
            ]
            | None
        ) = "random",
        k_best: int = 1,
        sigma: float = 10,
        base_estimator: BaseEstimator | None = None,
    ) -> None:
        """
        Initialize the class based on the y imputation methods.

        Args:
            y_imputation_method (Literal['random', 'naive', 'pseudo_labels',
                'self_training', 'label_propagation'] | None, optional):
                Y imputation methods to be used:
                - 'random': missing y is drawn from the binomial distribution (default);
                - 'naive': missing y and corresponding records are simply omitted.
                - 'pseudo_labels'
                - 'self_training'
                - 'label_propagation'
            k_best (int): Number of biggest probabilties to be selected in self_training method.
                Defaults to 1.
            sigma (float): Sigma paramater for the distance function in label_propagation method.
                Deafults to 1.0.
        """
        self.y_imputation_method = y_imputation_method
        self.model = LogisticLassoFistaCV()
        self.k_best = k_best
        self.sigma = sigma
        self.base_estimator = base_estimator

    def fit(
        self, X: ArrayLike, y_obs: ArrayLike, y_true: ArrayLike | None = None
    ) -> "UnlabeledLogReg":
        """
        Fit the model according to the given training data.

        Args:
            X (ArrayLike): Training matrix of size (n_samples, n_features).
            y_obs (ArrayLike): Target vector with potential missing values, relative to X.
            y_true (ArrayLike | None, optional): Target vector without missing values.
                Used to check the performance of label imputation. Defaults to None

        Returns:
            UnlabeledLogReg: Fitted estimator.
        """
        X = np.asarray(X)
        y_obs = np.asarray(y_obs)
        if y_true is not None:
            y_true = np.asarray(y_true)
            self.y_true = y_true
        self.imputation_scores = []

        X_complete, y_complete = self._impute(X, y_obs)  # Subset with not missing y

        self.model.fit(X_complete, y_complete)

        _imputation_methods = {
            "pseudo_labels": self._fit_pseudo_labels,
            "self_training": self._fit_self_training,
            "label_propagation": self._fit_label_propagation,
        }

        if self.y_imputation_method in _imputation_methods:
            _imputation_methods[self.y_imputation_method](
                X, y_obs, X_complete, y_complete
            )

        return self

    def _fit_pseudo_labels(
        self,
        X: ArrayLike,
        y_obs: ArrayLike,
        X_complete: ArrayLike,
        y_complete: ArrayLike,
    ) -> None:
        """Fit the model with a pseudo labels approach."""
        # train model on data without missing labels, predict them and train again
        y_pred = self.model.predict(self.X_original)

        y_all = self.y_original.copy()
        missing_mask = y_all == -1

        y_all[missing_mask] = y_pred[missing_mask]

        self.model.fit(self.X_original, y_all)

    def _fit_self_training(
        self,
        X: ArrayLike,
        y_obs: ArrayLike,
        X_complete: ArrayLike,
        y_complete: ArrayLike,
    ) -> None:
        """Fit the model with a self training approach."""
        # Base estimator
        if self.base_estimator is None:
            clf = clone(self.model)
        else:
            clf = clone(self.base_estimator)

        # Data without missing Y for training
        X_train = X_complete.copy()
        y_train = y_complete.copy()

        # X for missing Y
        X_missing = X[y_obs == -1]

        k = self.k_best

        missing_indices = np.where(y_obs == -1)[0]
        y_imputed = np.full(len(y_obs), -1)

        while len(X_missing) > 0:
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_missing)

            if len(X_missing) < 2 * k:
                y_pseudo_remaining = np.argmax(y_prob, axis=1)

                X_train = np.vstack([X_train, X_missing])
                y_train = np.concatenate((y_train, y_pseudo_remaining))
                break

            idx_class_0 = np.argsort(y_prob[:, 0])[-k:]
            idx_class_1 = np.argsort(y_prob[:, 1])[-k:]
            idx_to_add = np.union1d(idx_class_0, idx_class_1)

            X_train = np.vstack([X_train, X_missing[idx_to_add]])
            y_train = np.concatenate([y_train, np.argmax(y_prob[idx_to_add], axis=1)])
            X_missing = np.delete(X_missing, idx_to_add, axis=0)

            original_indices = missing_indices[idx_to_add]
            y_imputed[original_indices] = np.argmax(y_prob[idx_to_add], axis=1)

            missing_indices = np.delete(missing_indices, idx_to_add)

            self._calculate_imputation_performance(y_obs, y_imputed)

        self.model.fit(X_train, y_train)

    def _fit_label_propagation(
        self,
        X: ArrayLike,
        y_obs: ArrayLike,
        X_complete: ArrayLike,
        y_complete: ArrayLike,
    ) -> None:
        """Fit the model with a label propagation approach."""

        # knn graph with distances between points
        def find_minimal_k(X: ArrayLike):
            "Find minimal k for which the kNN graph is connected."
            k = 1
            while True:
                W = kneighbors_graph(X, n_neighbors=k, mode="distance").toarray()
                n_components, _ = connected_components(W)
                if n_components == 1:
                    break
                k += 1
            return k

        k = find_minimal_k(X)
        W = kneighbors_graph(X, n_neighbors=k, mode="distance").toarray()
        W = np.maximum(W, W.T)
        zero_mask = W == 0

        # gaussian kernel (distances -> similarities)
        W = np.exp((-(W**2)) / self.sigma**2)
        W[zero_mask] = 0

        # normalizing to create T (T = probabilities)
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        T = W / row_sums[:, np.newaxis]

        Y = self.y_original.copy()  # original labels
        F = self.y_original.copy()  # current labels
        F[F == -1] = 0
        F_prev = F.copy()

        while True:
            # neighbors labels are weighted with probability
            F = T @ F
            labeled_mask = self.y_original != -1
            F[labeled_mask] = Y[labeled_mask]  # original labels
            # if changes between iterations are small
            if np.max(np.abs(F - F_prev)) < 1e-3:
                break

            self._calculate_imputation_performance(y_obs, (F >= 0.5).astype(int))
            F_prev = F.copy()

        y_completed = (F >= 0.5).astype(int)
        self.model.fit(self.X_original, y_completed)

    def _calculate_imputation_performance(
        self, y_obs: ArrayLike, y_imput_curr: ArrayLike
    ) -> None:
        """Calculate balanced accuracy of iterative imputation method."""
        if not hasattr(self, "y_true"):
            return
        missing_mask = y_obs == -1
        y_true_missing = self.y_true[missing_mask]
        y_imput_missing = y_imput_curr[missing_mask]
        valid_mask = y_imput_missing != -1
        score = balanced_accuracy_score(
            y_true_missing[valid_mask], y_imput_missing[valid_mask]
        )
        self.imputation_scores.append(score)

    def validate(
        self, X: ArrayLike, y: ArrayLike, measure: str, prob_threshold: float = 0.5
    ) -> tuple[float, float]:
        """
        Validate a fitted model to choose the best parameters.

        Args:
            X (ArrayLike): Validation matrix of size (n_samples, n_features).
            y (ArrayLike): Target vector relative to X.
            measure (str): Measure to be used for validation (e.g., 'balanced_accuracy', 'f1', etc.).
            prob_threshold (float, optional): Threshold for classifying probabilities. Defaults to 0.5.

        Returns:
            tuple[float, float]: A tuple containing the best mean score and the best mean score with one standard error.
        """
        return self.model.validate(X, y, measure, prob_threshold)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the probability estimates for the given input data.

        Args:
            X (ArrayLike): Input data for which to predict probabilities.

        Returns:
            ArrayLike: The predicted probabilities for each class for the input samples.
        """
        return self.model.predict_proba(X)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the output for the given input data.

        Args:
            X (ArrayLike): Input data for which predictions are to be made.

        Returns:
            ArrayLike: Predicted output based on the input data.
        """
        return self.model.predict(X)

    def _impute(self, X: ArrayLike, y_obs: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Impute missing values in the target variable based on the specified imputation method.

        Args:
            X (ArrayLike): Training matrix of size (n_samples, n_features).
            y_obs (ArrayLike): Target vector with potential missing values, relative to X.

        Returns:
            tuple[ArrayLike, ArrayLike]: A tuple containing the complete feature matrix and the imputed target variable.
        """
        y_complete = y_obs.copy()
        X_complete = X.copy()
        if self.y_imputation_method == "naive":
            return self._naive_imputation(X_complete, y_complete)

        elif self.y_imputation_method == "random":
            return X_complete, self._random_imputation(y_complete)

        elif self.y_imputation_method in [
            "pseudo_labels",
            "self_training",
            "label_propagation",
        ]:
            return self._pseudo_labeling(X_complete, y_complete)

        else:
            raise ValueError("Not existing imputation method.")

    def _random_imputation(self, y: ArrayLike) -> ArrayLike:
        """
        Randomly impute missing values in the input array.

        Args:
            y (ArrayLike): An array containing the data, where missing values are represented by -1.

        Returns:
            ArrayLike: The input array with missing values imputed.
        """
        missing_mask = y == -1
        y[missing_mask] = np.random.binomial(n=1, p=0.5, size=missing_mask.sum())
        return y

    def _naive_imputation(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Remove records with missing y.

        Args:
            X (ArrayLike): Training matrix of size (n_samples, n_features).
            y (ArrayLike): An array containing the data, where missing values are represented by -1.

        Returns:
            tuple[ArrayLike, ArrayLike]: X and y without the records with missing y.
        """
        missing_mask = y == -1
        return X[~missing_mask], y[~missing_mask]

    def _pseudo_labeling(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Remove records with missing y, but save the original X and y.

        Args:
            X (ArrayLike): Training matrix of size (n_samples, n_features).
            y (ArrayLike): An array containing the data, where missing values are represented by -1.

        Returns:
            tuple[ArrayLike, ArrayLike]: X and y without the records with missing y.
        """
        self.X_original = np.asarray(X.copy())
        self.y_original = np.asarray(y.copy())

        missing_mask = y == -1
        return X[~missing_mask], y[~missing_mask]
