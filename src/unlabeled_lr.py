"""This file contains implementation of the UnlabeledLogReg class."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from fista import LogisticLassoFistaCV


class UnlabeledLogReg:
    """
    A class for semi-supervised logistic regression.
    """

    def __init__(
        self,
        y_imputation_method: (
            Literal["random", "naive", "pseudo_labels", "iterative_pseudo_labels"] | None
        ) = "random",
        imp_prob_threshold: float | None = 0.8,
    ) -> None:
        """
        Initialize the class based on the y imputation methods.

        Args:
            y_imputation_method (Literal['random', 'naive'] | None, optional):
                Y imputation methods to be used:
                - 'random': missing y is drawn from the binomial distribution (default);
                - 'naive': missing y and corresponding records are simply omitted.
        """
        self.y_imputation_method = y_imputation_method
        self.imp_prob_threshold = imp_prob_threshold
        self.model = LogisticLassoFistaCV()

    def fit(self, X: ArrayLike, y_obs: ArrayLike) -> "UnlabeledLogReg":
        """
        Fit the model according to the given training data.

        Args:
            X (ArrayLike): Training matrix of size (n_samples, n_features).
            y_obs (ArrayLike): Target vector with potential missing values, relative to X.

        Returns:
            UnlabeledLogReg: Fitted estimator.
        """
        X_complete, y_complete = self._impute(X, y_obs)

        self.model.fit(X_complete, y_complete)

        if self.y_imputation_method == "pseudo_labels":
            # train model on data without missing labels, predict them and train again
            y_pred = self.model.predict(self.X_original)

            y_all = self.y_original.copy()
            missing_mask = y_all == -1

            y_all[missing_mask] = y_pred[missing_mask]

            self.model.fit(self.X_original, y_all)

        elif self.y_imputation_method == "iterative_pseudo_labels":
            y_not_completed = self.y_original.copy()
            t = self.imp_prob_threshold

            while True:
                y_prob = self.model.predict_proba(self.X_original)

                mask = (y_not_completed == -1) & ((y_prob >= t) | (y_prob <= 1 - t))
                num_discovered = np.sum(mask > 0)
                print(f"Discovered {num_discovered} new confident y values")

                if num_discovered > 0:
                    y_not_completed[mask] = (y_prob[mask] >= 0.5).astype(int)

                    valid_mask = y_not_completed != -1
                    X_train_step = self.X_original[valid_mask]
                    y_train_step = y_not_completed[valid_mask]

                    self.model.fit(X_train_step, y_train_step)
                else:
                    print("Ending iterative process. Assigning remaining y with 0.5 threshold...")
                    final_mask = y_not_completed == -1

                    if np.sum(final_mask) > 0:
                        y_not_completed[final_mask] = (y_prob[final_mask] >= 0.5).astype(int)
                        self.model.fit(self.X_original, y_not_completed)
                    break

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

        elif self.y_imputation_method in ["pseudo_labels", "iterative_pseudo_labels"]:
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
        missing_mask = (y == -1).values
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

        missing_mask = (y == -1).values
        return X[~missing_mask], y[~missing_mask]
