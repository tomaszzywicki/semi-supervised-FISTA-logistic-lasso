import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Tranformer class that removes a feature if it is highly correlated with
    another feature.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        """
        Initilize the class based on the given correlation threshold.

        Args:
            threshold (float, optional): Correlation threshold, above which the
                fetaures are considered highly correlated. Defaults to 0.7.
        """
        super().__init__()
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnSelector":
        """
        Fit the transformer to the given data by saving columns to be dropped.

        Args:
            X (pd.DataFrame): Training DataFrame.

        Returns:
            ColumnSelector: Fitted transformer.
        """
        X = pd.DataFrame(X)
        corr_matrix = X.corr().abs()
        rows, cols = np.triu_indices(n=len(corr_matrix.columns), k=1)
        drop_cols = set()
        for r, c in zip(rows, cols):
            if corr_matrix.iloc[r, c] > self.threshold:
                drop_cols.add(corr_matrix.columns[c])
        self.cols_to_drop_ = drop_cols

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the given data by dropping columns.

        Args:
            X (pd.DataFrame): DataFrame in which the columns will be dropped.

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_, errors="ignore")
