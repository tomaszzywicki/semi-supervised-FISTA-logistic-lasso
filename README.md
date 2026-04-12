# Semi-Supervised Learning Imputation Framework

This repository provides a framework for evaluating semi-supervised learning (SSL) imputation approaches. It is designed to handle artificially introduced missing data under various missing data mechanisms. The core algorithm relies on an Unlabeled Logistic Regression model (optimized via FISTA) and several label imputation techniques.


## Supported Missing Data Mechanisms

The framework allows you to degrade your fully labeled datasets using the following mechanisms.

* **MCAR (Missing Completely At Random):** The probability of a label being missing is a constant `p` and is independent of any features or the target variable.
* **MAR1 (Missing At Random - Single Feature):** The missingness depends on a single, randomly selected feature using a logistic sigmoid function. Controlled by weight `w` and bias `b`.
* **MAR2 (Missing At Random - All Features):** The missingness depends on a linear combination of all available explanatory variables.
* **MNAR (Missing Not At Random):** The probability of missingness depends on both the observed features and the *unobserved* true target label. Controlled by feature weights `w_x`, target weight `w_y`, and bias `b`.

---

## Imputation and Modeling Approaches

When initializing the `UnlabeledLogReg` model, you can choose from the following strategies to handle the missing labels:

* `naive`: Simply drops all instances with missing labels and trains the model on the remaining labeled data.
* `pseudo_labels`: Trains a model on labeled data, predicts the missing labels once, and retrains the model on the combined dataset.
* `self_training`: Uses a chosen base estimator (e.g., Random Forest, SVM) to iteratively label the most confident missing instances (`k_best`) and add them to the training set.
* `label_propagation`: Uses a graph-based approach (RBF kernel with parameter $\sigma$) to propagate labels from known instances to unknown instances based on feature similarity.

---

## Requirements and Installation

To run the experiments, ensure you have Python 3.9+ installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost openml
```

## How to Run the Algorithm

Open `instructions_notebook.ipynb` in Jupyter. This notebook contains:

1. **Part 1:** A guide on how to instantiate the `UnlabeledLogReg` model, apply a missing data mechanism, and fit the model on a single dataset.
2. **Part 2:** A generalized template for running the full, multi-seed experiment on your own `.csv` data.
3. **Part 3:** An example using the DARWIN dataset.