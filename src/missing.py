import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.special import expit


def _add_missing_indicators(y: pd.DataFrame, mask: ArrayLike) -> pd.DataFrame:
    """
    Add missing indicators to the DataFrame.

    Args:
        y (pd.DataFrame): The DataFrame containing the data to be updated.
        mask (ArrayLike): A boolean array or mask indicating which rows in y are missing.

    Returns:
        pd.DataFrame: The updated DataFrame with missing indicators added.
    """

    y.loc[mask, "Y_observed"] = -1
    y.loc[mask, "S"] = 1
    y.loc[mask, "Missing_Y"] = "yes"

    return y


def MCAR(y: pd.DataFrame, p: float = 0.2) -> pd.DataFrame:
    """
    Introduce missing values into a DataFrame according to the
    Missing Completely At Random (MCAR) mechanism.

    Args:
        y (pd.DataFrame): The input DataFrame from which missing values will be introduced.
        p (float): The proportion of samples to be made missing. Default is 0.2.

    Returns:
        pd.DataFrame: A DataFrame with missing values introduced according to the MCAR mechanism.
    """

    y_missing = y.copy()

    mask = np.random.binomial(n=1, p=p, size=len(y)).astype(bool)

    return _add_missing_indicators(y_missing, mask)


def MAR1(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_column: str,
    w: float = 1.0,
    b: float = 0.0,
) -> pd.DataFrame:
    """
    Introduces missing values in the target variable y based on a logistic model
    that uses a specified feature from the input DataFrame X.
    Uses  Missing At Random (MAR) mechanism.

    Args:
        X (pd.DataFrame): The input features DataFrame containing the feature used
                          to determine the missingness.
        y (pd.DataFrame): The target variable DataFrame in which missing values
                          will be introduced.
        feature_column (str): The index of the column in X that will be used to
                              model the missingness.
        w (float, optional): The weight applied to the feature values in the logistic
                             function. Defaults to 1.0.
        b (float, optional): The bias added to the weighted feature values in the
                             logistic function. Defaults to 0.0.

    Returns:
        pd.DataFrame:  A DataFrame with missing values introduced according to the MAR mechanism.
    """

    y_missing = y.copy()

    feature_values = X[feature_column].values

    mean_val = np.mean(feature_values)
    std_val = np.std(feature_values)

    if std_val == 0:
        z = np.zeros_like(feature_values)
    else:
        z = (feature_values - mean_val) / std_val

    logit = w * z + b
    p_missing = expit(logit)

    mask = np.random.rand(len(y)) < p_missing

    return _add_missing_indicators(y_missing, mask)


def MAR2(
    X: pd.DataFrame,
    y: pd.DataFrame,
    W: ArrayLike | None = None,
    b: float = 0.0,
) -> pd.DataFrame:
    """
    Introduces missing values in the target variable y based on a logistic model
    parameterized by the features X and weights W.
    Uses  Missing At Random (MAR) mechanism.

    Args:
        X (pd.DataFrame): The input features DataFrame.
        y (pd.DataFrame): The target variable DataFrame in which missing values
                          will be introduced.
        W (ArrayLike, optional): The weights applied to the feature's values in
                          the logistic function. Defaults to None.
        b (float, optional): The bias added to the weighted feature's values in
                          the logistic function. Defaults to 0.0.

    Returns:
        pd.DataFrame: A DataFrame with missing values introduced according to the MAR mechanism.
    """

    y_missing = y.copy()

    means = X.mean(axis=0).to_numpy(copy=True)
    stds = X.std(axis=0).to_numpy(copy=True)
    stds[stds == 0] = 1.0

    X_scaled = (X.values - means) / stds

    if W is None:
        W = np.random.randn(X.shape[1])

    z = np.dot(X_scaled, W) + b
    p_missing = expit(z)

    mask = np.random.rand(len(y)) < p_missing

    return _add_missing_indicators(y_missing, mask)


def MNAR(
    X: pd.DataFrame,
    y: pd.DataFrame,
    w_x: ndarray | None = None,
    w_y: float = 1.0,
    b: float = 0.0,
) -> pd.DataFrame:
    """
    Generates missing data in a dataset based on a Missing Not At Random (MNAR) mechanism.

    Args:
        X (pd.DataFrame): The input features DataFrame.
        y (pd.DataFrame):  The target variable DataFrame in which missing values
                          will be introduced.
        w_x (float | None): Coefficients for the features in X. If None, random
                          coefficients are generated. Defaults to None.
        w_y (float): Coefficient for the true values in y. Defaults to 1.0.
        b (float): Intercept term for the logistic model. Defaults to 0.0.

    Returns:
        pd.DataFrame: A DataFrame with missing values introduced according to the MNAR mechanism.
    """

    y_missing = y.copy()

    X_stds = X.std(axis=0).to_numpy(copy=True)
    X_stds[X_stds == 0] = 1.0

    X_scaled = (X.to_numpy(copy=True) - X.mean(axis=0).to_numpy(copy=True)) / X_stds

    y_true = y_missing["Y_true_unobserved"].to_numpy(copy=True)
    y_std = np.std(y_true)

    if y_std == 0:
        y_std = 1.0

    y_scaled = (y_true - y_true.mean()) / y_std

    if w_x is None:
        w_x = np.random.randn(X.shape[1])

    z = np.dot(X_scaled, w_x) + (w_y * y_scaled) + b
    p_missing = expit(z)

    mask = np.random.rand(len(y)) < p_missing

    return _add_missing_indicators(y_missing, mask)
