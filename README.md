# Wine-Quality-Classification
Wine quality classification using data preprocessing, exploratory data analysis, feature engineering, and Scikit-learn machine learning models.

# Wine Quality Classification — Model Comparison

A comparative machine learning pipeline predicting wine quality scores using the [WineQT dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) from Kaggle. This project evaluates four modeling approaches — Logistic Regression, Random Forest (single split), Random Forest (Stratified K-Fold), and hyperparameter-tuned XGBoost — to compare how model complexity and validation strategy affect performance on an imbalanced, multi-class target.

## Problem

Wine quality scores (originally 3–8) are treated as classification targets rather than a continuous regression output, since scores represent discrete quality tiers. The dataset is imbalanced — most samples cluster around quality scores 5 and 6, with very few examples at the extremes (3, 4, 8).

## Approach

1. **Data Cleaning** — Verified no missing values via a null-value heatmap.
2. **Preprocessing** — Target labels shifted (`-3`) to zero-index classes for XGBoost compatibility. `StandardScaler` applied for Logistic Regression, since features span vastly different numeric ranges (e.g., citric acid ~10⁻², total sulfur dioxide ~10¹).
3. **Modeling** — Four models trained and compared:
   - Logistic Regression (scaled features, single train/test split)
   - Random Forest (single train/test split)
   - Random Forest (5-fold Stratified K-Fold cross-validation)
   - XGBoost (hyperparameter-tuned via RandomizedSearchCV, `scoring='roc_auc_ovr_weighted'`)
4. **Evaluation** — Confusion matrix, weighted precision, recall, and F1-score for each model (weighted averaging used due to class imbalance).

## Results

| Model | Precision | F1-Score | Recall |
|---|---|---|---|
| Logistic Regression | 0.592 | 0.601 | 0.618 |
| Random Forest (single split) | 0.642 | 0.643 | 0.656 |
| **Random Forest (Stratified K-Fold)** | **0.633** | **0.645** | **0.661** |
| XGBoost (RandomizedSearchCV tuned with scoring roc_auc_ovr_weighted) | 0.629 | 0.638 | 0.654 |

**Best model: Random Forest with Stratified K-Fold cross-validation.**

## Key Finding

The Random Forest models performed better than Logistic Regression and XGBoost on this dataset. The Random Forest with Stratified K-Fold achieved the highest F1-score (0.645) and recall (0.661) among the models tested.

XGBoost performed slightly below Random Forest even after hyperparameter tuning. I also tested XGBoost with both roc_auc_ovr_weighted and f1_weighted as the tuning metric, but both produced the same best parameters and the same output.

## Tech Stack

* **Python** — Main programming language
* **Pandas** — Data loading and preprocessing
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Seaborn** — Heatmaps and exploratory data analysis
* **Scikit-learn** — Preprocessing, Logistic Regression, Random Forest, Stratified K-Fold, evaluation, and hyperparameter tuning
* **XGBoost** — Gradient boosting classification and hyperparameter tuning
* **Jupyter Notebook** — Development and experimentation


## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy seaborn scikit-learn xgboost`
3. Download `WineQT.csv` from the Kaggle link above and place it in the project directory
4. Run the notebook cells sequentially

## What I'd Explore Next

- Investigate feature importance to identify which chemical properties most influence quality classification
- Address class imbalance directly via oversampling (SMOTE) for the minority quality scores
