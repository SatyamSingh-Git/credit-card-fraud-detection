# Advanced Credit Card Fraud Detection Pipeline

This repository contains a comprehensive pipeline for detecting fraudulent credit card transactions. The project addresses the critical challenge of extreme class imbalance and employs feature engineering and modeling techniques to build a robust and reliable detection system.

## Project Overview

The goal of this project is to build a machine learning model that can accurately identify fraudulent credit card transactions from a highly imbalanced dataset. The pipeline demonstrates a professional data science workflow, emphasizing best practices such as preventing data leakage, robust evaluation, and proper handling of imbalanced data.

## Dataset

The project uses the "Credit Card Fraud Detection" dataset, which is publicly available on Kaggle.

- **Features:** The dataset contains 28 anonymized features (`V1` to `V28` from PCA), a `Time` feature, and an `Amount` feature.
- **Target:** The `Class` column is the target variable, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.
- **Challenge:** The dataset is highly imbalanced, with fraudulent transactions making up a very small fraction of the total data (~0.17%).
- Dataset link (Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

Note: dataset added (download `creditcard.csv` and place it in the project root or update the notebook path).

## Methodological Pipeline

The pipeline is structured to follow strict data science best practices, ensuring the model's evaluation is unbiased and reliable.

### 1. Data Splitting
The dataset is **immediately split** into a training set (80%) and a test set (20%). This is the most crucial step to prevent any form of **data leakage**, ensuring the test set remains a true proxy for unseen, real-world data.

### 2. Exploratory Data Analysis (EDA)
All EDA is performed **exclusively on the training set**. This prevents "data snooping," where insights from the test set might unintentionally influence feature engineering or modeling decisions.

### 3. Feature Engineering
- **Cyclical Time Feature:** The `Time` feature is converted from raw seconds into a cyclical representation of the hour of the day. This is achieved by transforming the hour into two new features, `Hour_sin` and `Hour_cos`, which correctly capture the 24-hour cycle (i.e., hour 23 is close to hour 0).
- **Scaling:** The `Amount` feature is scaled using `RobustScaler`. This scaler is chosen over `StandardScaler` because it is robust to outliers, which are present in the transaction amount data. The scaler is **fitted only on the training data** and then used to transform both the train and test sets.

### 4. Handling Class Imbalance with SMOTE
- **SMOTE (Synthetic Minority Over-sampling Technique)** is used to address the severe class imbalance.
- To prevent data leakage during cross-validation, SMOTE is applied as the first step within an `imblearn.pipeline.Pipeline`. This ensures that in each fold of cross-validation, oversampling is applied only to the training portion of that fold, and the validation portion remains clean.

### 5. Model Training and Hyperparameter Tuning
- **Models Compared:** The pipeline compares three powerful tree-based models:
  - **XGBoost (Tuned)**
  - **LightGBM**
  - **Random Forest**
- **Hyperparameter Tuning:** `GridSearchCV` is used to perform an extensive search for the best hyperparameters for the XGBoost model.
- **Cross-Validation:** A `StratifiedKFold` (with `n_splits=10`) strategy is used within `GridSearchCV`. This ensures that the class distribution is preserved in each fold and provides a highly reliable estimate of the model's performance.

### 6. Evaluation
- **Primary Metric:** The **Area Under the Precision-Recall Curve (AUPRC)** is used as the primary evaluation metric. AUPRC is more informative than ROC AUC for highly imbalanced datasets.
- **Other Metrics:** A full `classification_report` (including precision, recall, f1-score) and a `confusion_matrix` are generated for the final evaluation on the held-out test set.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/madhyala-bharadwaj/credit-card-fraud-detection
    ````markdown
    # Credit Card Fraud Detection (imbalanced data)

    This repo contains code and a notebook for exploring and modeling the Kaggle "Credit Card Fraud Detection" dataset. The analysis focuses on handling extreme class imbalance and comparing sampling strategies and classifiers.

    ## What this project does (short)
    - Loads the public `creditcard.csv` dataset (PCA V1..V28, Time, Amount, Class).
    - Scales `Time` and `Amount`, inspects distributions and correlations.
    - Creates a balanced subsample (random undersample) and also uses SMOTE (oversampling) during CV.
    - Removes extreme outliers (IQR) on selected features to reduce noise.
    - Trains and compares classifiers (Logistic Regression, KNN, SVC, Decision Tree, Random Forest, and XGBoost). Also includes a simple neural network baseline.

    ## Key methods and best practices
    - Keep a held-out test set from the original data and never leak resampling into that test set.
    - Apply SMOTE (or NearMiss) inside an `imblearn.pipeline` during cross-validation to avoid data leakage.
    - Use stratified splits for preserving class ratios during CV.

    ## Models included
    - Logistic Regression
    - K-Nearest Neighbors
    - Support Vector Classifier
    - Decision Tree
    - Random Forest
    - XGBoost (added and tuned via randomized search)
    - Simple Keras neural network (for comparison)

    ## Main metrics
    - Precision, Recall, F1-score
    - ROC AUC and Precision-Recall AUC (useful for imbalanced data)

    ## How to run (Windows CMD)
    1. Put `creditcard.csv` in this repository root or update the path used in the notebook/script.
    2. Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```
    3. Run the notebook (recommended):
    ```powershell
    jupyter notebook credit-fraud-Detection-imbalanced-datasets.ipynb
    ```
    Or run the script (if present):
    ```powershell
    python creditCardFraudDetection.py
    ```

    ## Notes & next steps
    - XGBoost is included and configured to run with SMOTE in a pipeline; tune the randomized/grid search parameters to improve results.
    - The notebook contains visualization cells (heatmaps, boxplots, t-SNE/PCA) — run interactively for best inspection.

    ````
