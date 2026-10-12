"""
Hotel Booking Cancellation Prediction - Training Pipeline
=========================================================
Steps
  1. Import libraries
  2. Data loading
  3. Data preprocessing (null values, duplicates, leakage columns)
  4. Feature selection (filter method - mutual information)
  5. Categorical encoding (LabelEncoder)
  6. Train / test split
  7. Scaling (StandardScaler)
  8. PCA on scaled data
  9. Hyper-parameter tuning (RandomizedSearchCV) -> pick best classifier
 10. Final model training
 11. Model evaluation (train vs test)
 12. Save artifacts for predict.py

Run:  python train_model.py
"""

# ----------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------------------------------------
import time
import warnings
from functools import partial
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hotel_bookings.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "is_canceled"

# These columns reveal the outcome (or are only known after the booking
# decision), so using them would give a fake, near-perfect score.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type"]

K_FEATURES = 20          # number of features kept by the filter method
TEST_SIZE = 0.20
RANDOM_STATE = 42
PCA_VARIANCE = 0.95      # keep 95% of the variance
N_ITER = 10              # random-search iterations per model
CV_FOLDS = 3
SCORING = "f1"           # metric used to choose the best model


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# 2. DATA LOADING
# ----------------------------------------------------------------------------
section("2. DATA LOADING")
df = pd.read_csv(DATA_PATH)
print(f"Shape            : {df.shape}")
print(f"Cancellation rate: {df[TARGET].mean():.2%}")

# ----------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ----------------------------------------------------------------------------
section("3. DATA PREPROCESSING")

# 3a. Null values
nulls = df.isnull().sum()
print("Null values before cleaning:")
print(nulls[nulls > 0].to_string())

df["children"] = df["children"].fillna(0)                 # few rows, assume no children
df["country"] = df["country"].fillna(df["country"].mode()[0])  # most frequent country
df["agent"] = df["agent"].fillna(0)                       # 0 = no agent
df["company"] = df["company"].fillna(0)                   # 0 = no company
print(f"Null values after cleaning : {int(df.isnull().sum().sum())}")

# 3b. Duplicates
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows dropped     : {n_dup}")

# 3c. Leakage columns
df = df.drop(columns=LEAKAGE_COLS)
print(f"Leakage columns dropped    : {LEAKAGE_COLS}")

# 3d. Invalid records (no guests at all, negative price)
before = len(df)
df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
df = df[df["adr"] >= 0].reset_index(drop=True)
df["children"] = df["children"].astype(int)
print(f"Invalid rows dropped       : {before - len(df)}")
print(f"Final shape                : {df.shape}")

# ----------------------------------------------------------------------------
# 4. FEATURE SELECTION - FILTER METHOD (mutual information)
# ----------------------------------------------------------------------------
section("4. FEATURE SELECTION (FILTER METHOD)")
X_all = df.drop(columns=[TARGET])
y = df[TARGET]

cat_cols_all = X_all.select_dtypes(exclude="number").columns.tolist()

# temporary numeric copy only for scoring the features
X_tmp = X_all.copy()
for col in cat_cols_all:
    X_tmp[col] = X_tmp[col].astype("category").cat.codes
discrete_mask = np.array([c in cat_cols_all for c in X_tmp.columns])

selector = SelectKBest(
    score_func=partial(
        mutual_info_classif, discrete_features=discrete_mask, random_state=RANDOM_STATE
    ),
    k=K_FEATURES,
)
selector.fit(X_tmp, y)

scores = pd.Series(selector.scores_, index=X_tmp.columns).sort_values(ascending=False)
selected_features = scores.head(K_FEATURES).index.tolist()
print("Mutual-information scores (top 25):")
print(scores.head(25).round(4).to_string())
print(f"\nSelected {len(selected_features)} features:\n{selected_features}")

X = X_all[selected_features].copy()
cat_cols = [c for c in selected_features if c in cat_cols_all]

# ----------------------------------------------------------------------------
# 5. CATEGORICAL COLUMNS - LABEL ENCODER
# ----------------------------------------------------------------------------
section("5. LABEL ENCODING")
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
print(f"Label-encoded columns: {cat_cols}")

# default values (used by predict.py when the user presses Enter)
defaults = {}
for col in selected_features:
    if col in cat_cols:
        defaults[col] = label_encoders[col].inverse_transform([int(X[col].mode()[0])])[0]
    else:
        defaults[col] = float(X[col].median())

# ----------------------------------------------------------------------------
# 6. DATA SPLITTING
# ----------------------------------------------------------------------------
section("6. TRAIN / TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# ----------------------------------------------------------------------------
# 7. DATA SCALING (fit on train only -> no leakage)
# ----------------------------------------------------------------------------
section("7. SCALING")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler applied.")

# ----------------------------------------------------------------------------
# 8. PCA ON SCALED DATA
# ----------------------------------------------------------------------------
section("8. PCA")
pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(
    f"Components kept: {pca.n_components_} of {X_train_scaled.shape[1]} "
    f"(explained variance = {pca.explained_variance_ratio_.sum():.2%})"
)

# ----------------------------------------------------------------------------
# 9. HYPER-PARAMETER TUNING - RandomizedSearchCV
# ----------------------------------------------------------------------------
section("9. RANDOMIZED SEARCH CV")
search_space = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"C": loguniform(1e-3, 1e2), "class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": randint(4, 25),
            "min_samples_split": randint(2, 30),
            "min_samples_leaf": randint(1, 20),
            "class_weight": [None, "balanced"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": randint(100, 300),
            "max_depth": randint(8, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        },
    ),
    "Hist Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "learning_rate": uniform(0.03, 0.27),
            "max_iter": randint(100, 400),
            "max_depth": randint(3, 15),
            "min_samples_leaf": randint(10, 60),
            "l2_regularization": uniform(0, 2),
        },
    ),
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}
for name, (estimator, params) in search_space.items():
    start = time.time()
    search = RandomizedSearchCV(
        estimator,
        param_distributions=params,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train_pca, y_train)
    results[name] = search
    print(
        f"{name:<24} CV {SCORING}: {search.best_score_:.4f}   "
        f"({time.time() - start:.0f}s)"
    )

best_name = max(results, key=lambda n: results[n].best_score_)
best_params = results[best_name].best_params_
print(f"\nBest model      : {best_name}")
print(f"Best parameters : {best_params}")

# ----------------------------------------------------------------------------
# 10. FINAL MODEL TRAINING
# ----------------------------------------------------------------------------
section("10. MODEL TRAINING")
final_model = clone(results[best_name].best_estimator_)
final_model.fit(X_train_pca, y_train)
print(f"{best_name} trained on {X_train_pca.shape[0]} rows.")

# ----------------------------------------------------------------------------
# 11. MODEL EVALUATION (train vs test)
# ----------------------------------------------------------------------------
section("11. MODEL EVALUATION")


def evaluate(X_data, y_true):
    y_pred = final_model.predict(X_data)
    y_prob = final_model.predict_proba(X_data)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }, y_pred


train_metrics, _ = evaluate(X_train_pca, y_train)
test_metrics, y_test_pred = evaluate(X_test_pca, y_test)

summary = pd.DataFrame({"Train": train_metrics, "Test": test_metrics}).round(4)
summary["Gap (Train-Test)"] = (summary["Train"] - summary["Test"]).round(4)
print(summary.to_string())

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification report (test):")
print(classification_report(y_test, y_test_pred, target_names=["Not canceled", "Canceled"]))

if summary.loc["F1-score", "Gap (Train-Test)"] > 0.08:
    print("Note: train score is noticeably higher than test -> some overfitting.")

# plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred, display_labels=["Not canceled", "Canceled"], ax=axes[0], cmap="Blues"
)
axes[0].set_title("Confusion matrix (test)")
RocCurveDisplay.from_estimator(final_model, X_test_pca, y_test, ax=axes[1])
axes[1].set_title(f"ROC curve - {best_name}")
plt.tight_layout()
plt.savefig(MODEL_DIR / "evaluation.png", dpi=130)
plt.close()
print(f"\nPlot saved to {MODEL_DIR / 'evaluation.png'}")

# ----------------------------------------------------------------------------
# 12. SAVE ARTIFACTS
# ----------------------------------------------------------------------------
section("12. SAVING MODEL")
artifacts = {
    "model": final_model,
    "model_name": best_name,
    "scaler": scaler,
    "pca": pca,
    "label_encoders": label_encoders,
    "selected_features": selected_features,
    "categorical_features": cat_cols,
    "defaults": defaults,
    "test_metrics": test_metrics,
}
joblib.dump(artifacts, MODEL_DIR / "hotel_cancellation_model.joblib")
print(f"Saved: {MODEL_DIR / 'hotel_cancellation_model.joblib'}")
print("\nDone. Now run:  python predict.py")
