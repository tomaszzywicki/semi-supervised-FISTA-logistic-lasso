# Semi-Supervised Learning Imputation Framework

This repository provides a comprehensive experimental framework for evaluating semi-supervised learning (SSL) imputation approaches. It is designed to handle artificially introduced missing data under various, highly customizable missing data mechanisms. The core algorithm relies on an Unlabeled Logistic Regression model (optimized via FISTA) and several label-inferring techniques.

## ✨ Key Features

1. **Custom Missing Data Generators:** Precisely simulate different real-world scenarios where labels are missing based on features, the target itself, or entirely at random.
2. **Multiple Imputation Strategies:** Compare baseline approaches (ignoring missing data) with advanced semi-supervised methods like Self-Training and Label Propagation.
3. **Automated Experimental Pipeline:** Easily run large-scale benchmarks across multiple random seeds, parameter configurations, and missing data mechanisms.

---

## 🛠️ Supported Missing Data Mechanisms

The framework allows you to degrade your fully labeled datasets using the following mechanisms. Missing labels are encoded as `-1` in the `Y_observed` column.

* **MCAR (Missing Completely At Random):** The probability of a label being missing is a constant `p` and is independent of any features or the target variable.
* **MAR1 (Missing At Random - Single Feature):** The missingness is driven by a single, randomly selected feature using a logistic sigmoid function. Controlled by weight `w` and bias `b`.
* **MAR2 (Missing At Random - All Features):** The missingness depends on a linear combination of all available explanatory variables.
* **MNAR (Missing Not At Random):** The most complex scenario. The probability of missingness depends on both the observed features and the *unobserved* true target label. Controlled by feature weights `w_x`, target weight `w_y`, and bias `b`.

---

## 🧠 Imputation and Modeling Approaches

When initializing the `UnlabeledLogReg` model, you can choose from the following strategies to handle the missing labels (`-1`):

* `naive`: Simply drops all instances with missing labels and trains the model on the remaining labeled data (acts as the baseline).
* `pseudo_labels`: Trains a model on labeled data, predicts the missing labels once, and retrains the model on the combined dataset.
* `iterative_pseudo_labels`: Iteratively refines the pseudo-labels until convergence before the final model fit.
* `self_training`: Uses a chosen base estimator (e.g., Random Forest, SVM) to iteratively label the most confident missing instances (`k_best`) and add them to the training set.
* `label_propagation`: Uses a graph-based approach (RBF kernel with parameter $\sigma$) to propagate labels from known instances to unknown instances based on feature similarity.

---

## 📦 Requirements and Installation

To run the experiments, ensure you have Python 3.9+ installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost openml
```
## 📂 Project Structure

* `data/` - Datasets used in the experiments.
  * `raw/` - Original, unmodified datasets (e.g., DARWIN, LSVT, Parkinson, Prostate, Sonar).
  * `processed/` - Cleaned and formatted datasets ready for the pipeline.
* `notebooks/` - Jupyter notebooks for exploration, tutorials, and plotting.
  * `00_datasets_check.ipynb` - Initial data exploration and validation.
  * `01_data_prep.ipynb` - Step-by-step data preprocessing.
  * `02_fista_sklearn_comparison.ipynb` - Validation of the custom FISTA implementation against standard scikit-learn models.
  * `experiments.ipynb` - Main workspace for running the large-scale benchmark experiments.
  * `instructions_notebook.ipynb` - Step-by-step tutorial on how to use the framework.
  * `plots.ipynb` - Dedicated notebook for generating and saving report visualizations.
* `results/` - Output directory for experimental metrics and charts.
  * `label_propagation/` - Specific results and sub-experiments for the label propagation $\sigma$ analysis.
  * `results_*.csv` - Aggregated evaluation metrics for respective datasets.
* `src/` - Core Python source code.
  * `fista.py` - Implementation of the custom `LogisticLassoFistaCV` optimizer.
  * `unlabeled_lr.py` - The `UnlabeledLogReg` class implementing semi-supervised strategies (pseudo-labeling, self-training, label propagation).
  * `experiment.py` - Main execution loops (`run_experiment`, `run_label_propagation_experiment`).
  * `missing.py` - Data degradation functions simulating MCAR, MAR1, MAR2, and MNAR mechanisms.
  * `utils.py` - Mathematical helpers (gradients, sigmoid, FISTA steps), metrics evaluation, and ARFF loaders.
  * `visualizations.py` - Tools for generating boxplots, summary tables, and $\sigma$ analysis charts.
  * `data_prep.py` - Scikit-learn compatible preprocessing utilities (e.g., `ColumnSelector`).

## 🚀 How to Run the Algorithm

Open `instructions_notebook.ipynb` in Jupyter. This notebook contains:

1. **Part 1:** A step-by-step guide on how to instantiate the `UnlabeledLogReg` model, apply a missing data mechanism, and fit the model on a single dataset.
2. **Part 2:** A generalized template for running the full, multi-seed experimental suite on your own `.csv` data.
3. **Part 3:** A fully reproducible example using the DARWIN dataset.