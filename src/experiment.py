import logging

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_prep import ColumnSelector
from missing import *
from unlabeled_lr import UnlabeledLogReg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment(
    X: ArrayLike | pd.DataFrame,
    y: ArrayLike | pd.Series | pd.DataFrame,
    mnar_w: float,
    mar1_w: float,
    mar1_b: float,
    seeds: list[int],
    approaches: list[str],
    k_best: list[int],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Executes an experiment to evaluate different semi-supervised learning
    approaches for handling missing labels under various missing data mechanisms.

    The function splits the dataset into train, validation, and test sets, applies
    a preprocessing pipeline, and artificially removes labels based on MCAR, MAR1,
    MAR2, and MNAR schemes. It then trains an UnlabeledLogReg model using the
    specified approaches and compares their performance against an Oracle baseline.

    Args:
        X (ArrayLike or pd.DataFrame): The input features dataset.
        y (ArrayLike or pd.Series): The binary target variable.
        mnar_w (float): The weight applied to the true label (Y) in the MNAR scheme.
        mar1_w (float): The weight applied to the randomly selected feature in the MAR1 scheme.
        mar1_b (float): The bias (intercept) term used in the MAR1 scheme.
        seeds (list[int]): A list of random seeds for reproducible experimental runs.
        approaches (list[str]): A list of string identifiers for the imputation
            methods to evaluate (e.g., ['naive', 'pseudo_labels']).

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics (Accuracy,
            Balanced_Acc, F1, ROC_AUC) and the percentage of missing data
            for each seed, scheme, and approach.
    """

    y = np.asarray(y).ravel()
    results = []

    for seed in seeds:

        np.random.seed(seed)
        logger.info(f"Experiment for SEED: {seed}")

        # Train (60%), Val (20%), Test (20%)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
        )

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", ColumnSelector(threshold=0.7)),
            ]
        )

        X_train = pipeline.fit_transform(X_train, y_train)
        X_val = pipeline.transform(X_val)
        X_test = pipeline.transform(X_test)

        X_train_df = pd.DataFrame(X_train)

        y_train_df = pd.DataFrame(y_train, columns=["Y_true_unobserved"])
        y_train_without_missing = y_train_df["Y_true_unobserved"].copy()

        y_train_df["Y_observed"] = y_train_df["Y_true_unobserved"].copy().astype(int)
        y_train_df["S"] = 0
        y_train_df["Missing_Y"] = "no"

        random_col_name = np.random.choice(X_train_df.columns)
        schemes_config = [
            {"name": "MCAR_0.2", "type": "MCAR", "params": {"p": 0.2}},
            {"name": "MCAR_0.5", "type": "MCAR", "params": {"p": 0.5}},
            {"name": "MCAR_0.8", "type": "MCAR", "params": {"p": 0.8}},
            {
                "name": "MAR1",
                "type": "MAR1",
                "params": {"w": mar1_w, "b": mar1_b, "feature_column": random_col_name},
            },
            {"name": "MAR2", "type": "MAR2", "params": {}},
            {"name": "MNAR", "type": "MNAR", "params": {"w_y": mnar_w}},
        ]

        # oracle
        if verbose:
            logger.info("Training approach ORACLE...")
        oracle_model = UnlabeledLogReg(y_imputation_method="naive")
        oracle_model.fit(X_train, y_train_without_missing)

        oracle_model.validate(X_val, y_val, measure="balanced_accuracy")

        y_pred_oracle = oracle_model.predict(X_test)
        y_prob_oracle = oracle_model.predict_proba(X_test)

        results.append(
            {
                "Seed": seed,
                "Scheme": "None",
                "Approach": "Oracle",
                "k_best": np.nan,
                "base_estimator": np.nan,
                "Missing_Percent": 0.0,
                "Accuracy": accuracy_score(y_test, y_pred_oracle),
                "Balanced_Acc": balanced_accuracy_score(y_test, y_pred_oracle),
                "F1": f1_score(y_test, y_pred_oracle),
                "ROC_AUC": roc_auc_score(y_test, y_prob_oracle),
                "Imputation_score": []
            }
        )

        # iteration over missing mechanisms
        for config in schemes_config:
            scheme_name = config["name"]
            scheme_type = config["type"]

            if verbose:
                logger.info(f"Testing scheme: {scheme_name}")

            if scheme_type == "MCAR":
                y_missing_df = MCAR(y_train_df, **config["params"])
            elif scheme_type == "MAR1":
                y_missing_df = MAR1(
                    pd.DataFrame(X_train), y_train_df, **config["params"]
                )
            elif scheme_type == "MAR2":
                y_missing_df = MAR2(
                    pd.DataFrame(X_train), y_train_df, **config["params"]
                )
            elif scheme_type == "MNAR":
                y_missing_df = MNAR(
                    pd.DataFrame(X_train), y_train_df, **config["params"]
                )

            y_train_obs = y_missing_df["Y_observed"]

            # useful printing % of missing y info
            missing_count = (y_train_obs == -1).sum()
            total_count = len(y_train_obs)
            missing_pct = round((missing_count / total_count) * 100, 2)

            if verbose:
                logger.info(
                    f"Deleted {missing_pct}% y info ({missing_count}/{total_count})"
                )

            # iteration over approaches
            for approach in approaches:

                if verbose:
                    logger.info(f"Approach: {approach}")

                if approach == "self_training":
                    classifiers = [
                        SVC(C=1.0, kernel="rbf", probability=True),
                        LogisticRegression(l1_ratio=0, C=10.0, max_iter=10_000),
                        RandomForestClassifier(),
                        XGBClassifier(),
                    ]

                    for classifier in classifiers:
                        for k in k_best:

                            ulr_model = UnlabeledLogReg(
                                y_imputation_method=approach,
                                k_best=k,
                                base_estimator=classifier,
                            )
                            ulr_model.fit(X_train, y_train_obs, y_train_df["Y_true_unobserved"])

                            # valdiation
                            ulr_model.validate(
                                X_val, y_val, measure="balanced_accuracy"
                            )

                            # calculating performance on outer test data
                            y_pred = ulr_model.predict(X_test)
                            y_prob = ulr_model.predict_proba(X_test)

                            results.append(
                                {
                                    "Seed": seed,
                                    "Scheme": scheme_name,
                                    "Approach": approach,
                                    "k_best": k,
                                    "base_estimator": classifier.__class__.__name__,
                                    "Missing_Percent": missing_pct,
                                    "Accuracy": accuracy_score(y_test, y_pred),
                                    "Balanced_Acc": balanced_accuracy_score(
                                        y_test, y_pred
                                    ),
                                    "F1": f1_score(y_test, y_pred),
                                    "ROC_AUC": roc_auc_score(y_test, y_prob),
                                    "Imputation_score": ulr_model.imputation_scores
                                }
                            )

                    continue

                ulr_model = UnlabeledLogReg(y_imputation_method=approach)
                ulr_model.fit(X_train, y_train_obs, y_train_df["Y_true_unobserved"])

                # valdiation
                ulr_model.validate(X_val, y_val, measure="balanced_accuracy")

                # calculating performance on outer test data
                y_pred = ulr_model.predict(X_test)
                y_prob = ulr_model.predict_proba(X_test)

                results.append(
                    {
                        "Seed": seed,
                        "Scheme": scheme_name,
                        "Approach": approach,
                        "k_best": np.nan,
                        "base_estimator": np.nan,
                        "Missing_Percent": missing_pct,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Balanced_Acc": balanced_accuracy_score(y_test, y_pred),
                        "F1": f1_score(y_test, y_pred),
                        "ROC_AUC": roc_auc_score(y_test, y_prob),
                        "Imputation_score": ulr_model.imputation_scores
                    }
                )

    results_df = pd.DataFrame(results)
    logger.info("End of experiment")

    return results_df
