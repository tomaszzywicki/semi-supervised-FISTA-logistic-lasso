"""This file contains implementation of the UnlabeledLogReg class."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph

from xgboost import XGBClassifier

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
                "self_trainingg",
                "label_propagation",
            ]
            | None
        ) = "random",
        k_best: int = 1,
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
        """
        self.y_imputation_method = y_imputation_method
        self.model = LogisticLassoFistaCV()
        self.k_best = k_best

    def fit(self, X: ArrayLike, y_obs: ArrayLike) -> "UnlabeledLogReg":
        """
        Fit the model according to the given training data.

        Args:
            X (ArrayLike): Training matrix of size (n_samples, n_features).
            y_obs (ArrayLike): Target vector with potential missing values, relative to X.

        Returns:
            UnlabeledLogReg: Fitted estimator.
        """
        X = np.asarray(X)
        y_obs = np.asarray(y_obs)

        X_complete, y_complete = self._impute(X, y_obs)  # Subset with not missing y

        self.model.fit(X_complete, y_complete)

        if self.y_imputation_method == "pseudo_labels":
            # train model on data without missing labels, predict them and train again
            y_pred = self.model.predict(self.X_original)

            y_all = self.y_original.copy()
            missing_mask = y_all == -1

            y_all[missing_mask] = y_pred[missing_mask]

            self.model.fit(self.X_original, y_all)

        elif self.y_imputation_method == "self_training":
            k = self.k_best

            # Trzeba potem podać classifier
            # clf = XGBClassifier(
            #     eta=0.3,
            #     max_depth=6,
            # )
            clf = LogisticRegression(l1_ratio=0, C=10.0, max_iter=10_000)

            X_train_step = X_complete
            y_train_step = y_complete

            X_missing = X[y_obs == -1]

            while len(X_missing) > 0:
                clf.fit(X_train_step, y_train_step)

                if len(X_missing) < 2 * k:
                    y_prob = clf.predict_proba(X_missing)
                    y_pseudo_remaining = np.argmax(y_prob, axis=1)

                    X_train_step = np.vstack([X_train_step, X_missing])
                    y_train_step = np.concatenate((y_train_step, y_pseudo_remaining))
                    break

                y_prob = clf.predict_proba(X_missing)
                p0 = y_prob[:, 0]
                p1 = y_prob[:, 1]

                idx_class_0 = np.argsort(p0)[-k:]
                idx_class_1 = np.argsort(p1)[-k:]

                X_pseudo_0 = X_missing[idx_class_0]
                X_pseudo_1 = X_missing[idx_class_1]

                y_pseudo_0 = np.zeros(k, dtype=int)
                y_pseudo_1 = np.ones(k, dtype=int)

                X_top_k = np.vstack((X_pseudo_0, X_pseudo_1))
                y_pseudo_k = np.concatenate((y_pseudo_0, y_pseudo_1))

                X_train_step = np.vstack([X_train_step, X_top_k])
                y_train_step = np.concatenate((y_train_step, y_pseudo_k))

                idx_to_remove = np.concatenate(([idx_class_0, idx_class_1]))
                X_missing = np.delete(X_missing, idx_to_remove, axis=0)

            self.model.fit(X_train_step, y_train_step)
            return self

        elif self.y_imputation_method == "label_propagattion":

            # knn graph with distances between points
            W = kneighbors_graph(X, n_neighbors=5, mode="distance").toarray()
            zero_mask = W == 0

            # gaussian kernel (distances -> similarities)
            sigma = 1.0
            W = np.exp((-(W**2)) / sigma**2)
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
                if np.max(np.abs(F - F_prev)) < 1e-5:
                    break

                F_prev = F.copy()

            y_completed = F.copy()
            self.model.fit(self.X_original, y_completed)

        return self

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

    def _naive_imputation(self, X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
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

    def _pseudo_labeling(self, X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
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
