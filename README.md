# ☄️ Planetary Defense Analytics
**Classifying, Predicting and Clustering Asteroids with Machine Learning**

🔗 **Live app:** _<paste your Streamlit Cloud link here>_

## Overview
Machine learning on the NASA JPL Small-Body Database (958,524 asteroids × 45 columns) to support planetary-defense triage:

| Case | Task | Target / Method |
|---|---|---|
| 1 | Classification | Predict Potentially Hazardous Asteroid (`pha`) status |
| 2 | Regression | Predict asteroid `diameter` (km) |
| 3 | Clustering | Group asteroids by orbital elements (K-Means, DBSCAN) |

All three cases are completed, with MLflow experiment tracking, SHAP explainability and a deployed Streamlit app.

## Key decisions
- **Leakage control (Case 1):** PHA is defined by MOID ≤ 0.05 AU and H ≤ 22, so `moid`, `moid_ld`, `q`, `ad`, `neo` and `class` are excluded from the classifier.
- **Class imbalance (Case 1):** only ~0.22% of asteroids are PHAs. Compared baseline vs class weighting vs SMOTE; models are judged on Recall, Precision, F1 and PR-AUC, not accuracy.
- **Target transform (Case 2):** diameter is modelled as log10(diameter) because of extreme skew; diameter is never imputed.
- **Clustering (Case 3):** log-transformed and standardised `a, e, i, per, moid_ld`; k chosen by elbow + silhouette; DBSCAN eps tuned with a k-distance graph.

## Results
| Case | Best model | Key metrics |
|---|---|---|
| 1 — Classification | _fill in_ | Recall _ / Precision _ / F1 _ / PR-AUC _ |
| 2 — Regression | _fill in_ | R² _ / RMSE _ / MAE _ |
| 3 — Clustering | K-Means (k = _) | Silhouette _ |

## Project structure
```
planetary-defense-analytics/
├── app/app.py                     # Streamlit app
├── models/                        # Saved models used by the app
│   ├── pha_classifier.joblib
│   ├── diameter_regressor.joblib
│   └── cluster_model.joblib
├── notebooks/
│   ├── 01_eda_cleaning.ipynb      # Data understanding, cleaning, EDA
│   ├── 02_classification.ipynb    # Case 1 + feature selection + MLflow + SHAP
│   ├── 03_regression.ipynb        # Case 2 + feature selection + MLflow + SHAP
│   └── 04_clustering.ipynb        # Case 3: K-Means vs DBSCAN
├── requirements.txt
└── README.md
```

## Dataset
NASA JPL Small-Body Database export (Kaggle: "Asteroid Dataset"). Not included in the repo because of its size. Download the CSV and save it as `data/dataset.csv`.

## Run locally
```bash
git clone <your-repo-url>
cd planetary-defense-analytics
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```
To re-train, also install `jupyter mlflow pyarrow seaborn`, then run the notebooks in order (01 → 04).

## MLflow
Experiment runs for Case 1 and Case 2 are logged locally. To view them:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://127.0.0.1:5000.

## Tech stack
Python · pandas · scikit-learn · imbalanced-learn · XGBoost · SHAP · MLflow · Matplotlib / Seaborn · Streamlit · GitHub
