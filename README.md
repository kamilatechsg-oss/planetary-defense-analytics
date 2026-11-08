# ☄️ Planetary Defense Analytics
**Classifying, Predicting and Clustering Asteroids with Machine Learning**

🔗 **Live app:** https://kamila-planetary-defense.streamlit.app/ 
📂 **Repository:** https://github.com/kamilatechsg-oss/planetary-defense-analytics

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
- **Class imbalance (Case 1):** only ~0.22% of asteroids are PHAs (about 1 in 453). Baseline, class weighting and SMOTE were compared; models are judged on Recall, Precision, F1 and PR-AUC, not accuracy (predicting "not hazardous" for everything would score 99.8% accuracy).
- **Feature selection:** consensus of 3 methods (statistical tests, Random Forest importance, RFE); a feature is kept if at least 2 methods select it.
- **Target transform (Case 2):** diameter is modelled as log10(diameter) because of extreme skew (metres up to ~940 km); diameter is never imputed.
- **Clustering (Case 3):** log-transformed and standardised `a, e, i, per, moid_ld`; k chosen by elbow + silhouette; DBSCAN eps tuned with a k-distance graph.

## Results
| Case | Best model | Key metrics |
|---|---|---|
| 1 — Classification | XGBoost (SMOTE) | Recall 0.879 / Precision 0.369 / F1 0.520 / PR-AUC 0.561 |
| 2 — Regression | XGBoost | R² 0.973 / RMSE 0.046 (log10) / MAE 0.55 km |
| 3 — Clustering | K-Means (k = 5) | Silhouette 0.308 |

**Compared with the reference benchmark:** Case 1 recall is similar (0.88 vs 0.89) with more than 2× higher precision (0.37 vs 0.16) and F1 (0.52 vs 0.27); Case 2 R² is higher (0.973 vs 0.96); Case 3 silhouette is close (0.308 vs ≈0.35).

**Deployment notes:**
- **Case 1:** XGBoost with SMOTE has the best F1 and PR-AUC among the imbalance-aware models. Recall is prioritised because a missed PHA costs far more than a false alarm.
- **Case 2:** Random Forest had a slightly better RMSE (0.045), but its 119 MB file exceeds GitHub's 100 MB limit; XGBoost (0.8 MB) is deployed with almost identical accuracy.
- **Case 3:** k = 2 had the highest silhouette (0.87) but only separated a few distant outliers; k = 5 was chosen at the elbow point for astronomically meaningful clusters.

## Streamlit app
Enter an asteroid's orbital and physical parameters (or load an example) to see:
- **Hazard (PHA)** prediction with probability, adjustable decision threshold and a SHAP waterfall explanation
- **Predicted diameter** with a SHAP explanation and a physics-formula check
- **Cluster assignment** shown on a plot of the catalogued asteroid population

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
git clone https://github.com/kamilatechsg-oss/planetary-defense-analytics.git
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

## Limitations
- Expensive analysis steps (EDA plots, feature selection, silhouette, DBSCAN) use random samples; final models are trained and evaluated on the full train/test data.
- The diameter model uses some orbital features (e.g., MOID) that reflect survey/discovery bias rather than physics.
- Orbit-uncertainty features (`sigma_*`) partly reflect how well an object has been observed.

## Tech stack
Python · pandas · scikit-learn · imbalanced-learn · XGBoost · SHAP · MLflow · Matplotlib / Seaborn · Streamlit · GitHub