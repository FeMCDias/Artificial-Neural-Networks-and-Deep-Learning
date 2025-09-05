#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activity — Data Generation, Visualization, and Preprocessing
Author: Felipe Maluli de Carvalho Dias
Date: 05/09/2025

Requirements:
  pip install -r ../../requirements.txt

This script reproduces three exercises:
  1) 2D Gaussian data + MLP and decision-boundary visualization
  2) 5D data (different covariances) + 2D PCA
  3) Spaceship Titanic (Kaggle) preprocessing for tanh-based networks

Expected files for Exercise 3 (Available in ./data):
  - train.csv
  - test.csv
  - sample_submission.csv

Usage
-----
    python solution_exercises.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------
# Global configuration
# ---------------------------
RANDOM_STATE = 42                                       # reproducibility seed
rng = np.random.default_rng(RANDOM_STATE)               # NumPy random generator (modern API)
outdir = Path("./assets")                               # where figures/CSVs are saved
outdir.mkdir(parents=True, exist_ok=True)

# ===========================
# Exercise 1 — Synthetic 2D data + MLP decision regions
# ===========================
n_per_class = 100
# 4 cluster centers (corners of a square), all share the same isotropic covariance
means = {
    0: np.array([-2.0, -2.0]),
    1: np.array([ 2.0, -2.0]),
    2: np.array([ 2.0,  2.0]),
    3: np.array([-2.0,  2.0]),
}
sigma = 0.9
cov = (sigma**2) * np.eye(2)                            # isotropic 2D Gaussian (σ² I)

# --- sample 2D Gaussians per class ---
X_list, y_list = [], []
for c, mu in means.items():
    Xc = rng.multivariate_normal(mu, cov, size=n_per_class)  # draw N(mu, cov)
    yc = np.full(n_per_class, c, dtype=int)                  # class labels
    X_list.append(Xc); y_list.append(yc)

X1 = np.vstack(X_list)                                  # (N, 2) features
y1 = np.concatenate(y_list)                             # (N,) labels in {0,1,2,3}

# --- scatter plot of the four Gaussian blobs ---
plt.figure(figsize=(6, 6))
for c in sorted(means.keys()):
    mask = (y1 == c)
    plt.scatter(X1[mask, 0], X1[mask, 1], s=12, label=f"Class {c}")
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Ex.1 — 2D Scatter (4 Gaussian classes)")
plt.axis("equal"); plt.tight_layout()
plt.savefig(outdir/"exercise1_scatter.png", dpi=150)

# --- train a small MLP (tanh activations) directly on the 2D inputs ---
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 16),                        # two hidden layers
    activation="tanh",                                  # nonlinearity
    random_state=RANDOM_STATE,
    max_iter=2000                                       # generous cap to ensure convergence
).fit(X1, y1)

# --- draw decision regions by predicting over a grid ---
x_min, x_max = X1[:,0].min()-1.5, X1[:,0].max()+1.5
y_min, y_max = X1[:,1].min()-1.5, X1[:,1].max()+1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.25)                    # color regions by predicted class
for c in sorted(means.keys()):
    mask = (y1 == c)
    plt.scatter(X1[mask, 0], X1[mask, 1], s=10, label=f"Class {c}")
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Ex.1 — Decision regions (tanh MLP)")
plt.axis("equal"); plt.tight_layout()
plt.savefig(outdir/"exercise1_regions.png", dpi=150)

# ===========================
# Exercise 2 — 5D data with different covariances + PCA to 2D
# ===========================
# Two multivariate Gaussians (A, B) in 5D with different covariance structures
mu_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
Sigma_A = np.array([[1.5, 0.3, 0.0, 0.2, 0.0],
                    [0.3, 0.7, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 1.0, 0.2, 0.1],
                    [0.2, 0.0, 0.2, 0.8, 0.0],
                    [0.0, 0.0, 0.1, 0.0, 0.5]])
mu_B = np.array([0.6, 0.4, -0.6, 0.6, -0.4])
Sigma_B = np.array([[0.9, -0.2, 0.0, 0.0, 0.0],
                    [-0.2, 1.2, 0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.6, 0.1, 0.0],
                    [0.0, 0.0, 0.1, 1.1, 0.2],
                    [0.0, 0.0, 0.0, 0.2, 0.7]])
# Make sure covariances are exactly symmetric (good practice for MVN sampling)
Sigma_A = (Sigma_A + Sigma_A.T)/2
Sigma_B = (Sigma_B + Sigma_B.T)/2

# --- sample 5D Gaussians for each class ---
XA = rng.multivariate_normal(mu_A, Sigma_A, size=500)  # class A
XB = rng.multivariate_normal(mu_B, Sigma_B, size=500)  # class B
X2 = np.vstack([XA, XB])                                # (1000, 5)
y2 = np.array([0]*500 + [1]*500)                        # 0=A, 1=B

# --- dimensionality reduction to 2D with PCA for visualization ---
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X2_pca = pca.fit_transform(X2)

# --- scatter the projected classes (PC1 vs PC2) ---
plt.figure(figsize=(6, 6))
plt.scatter(X2_pca[y2==0, 0], X2_pca[y2==0, 1], s=10, label="Class A")
plt.scatter(X2_pca[y2==1, 0], X2_pca[y2==1, 1], s=10, label="Class B")
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Ex.2 — PCA (5D→2D)")
plt.tight_layout()
plt.savefig(outdir/"exercise2_pca_scatter.png", dpi=150)

# ===========================
# Exercise 3 — Spaceship Titanic preprocessing (for tanh-friendly inputs)
# ===========================
# CSVs expected under ./data (Kaggle dataset)
train_path = Path("./data/train.csv")
test_path  = Path("./data/test.csv")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

def split_cabin_fix(df):
    """
    Split 'Cabin' column (format 'Deck/Num/Side') into:
      - 'Deck' (str), 'CabinNum' (float), 'Side' (str)
    Handles missing or malformed values defensively.
    """
    deck, num, side = [], [], []
    for v in df["Cabin"]:
        if pd.isna(v):
            deck.append(np.nan); num.append(np.nan); side.append(np.nan)
        else:
            s = str(v); parts = s.split("/")
            if len(parts) == 3:
                d, n, sside = parts
                deck.append(d if d != "" else np.nan)
                try:
                    num.append(float(n))
                except:
                    num.append(np.nan)
                side.append(sside if sside != "" else np.nan)
            else:
                deck.append(np.nan); num.append(np.nan); side.append(np.nan)
    df = df.copy()
    df["Deck"] = deck; df["CabinNum"] = num; df["Side"] = side
    return df

# --- engineer Deck/CabinNum/Side from Cabin in both splits ---
train_df = split_cabin_fix(train_df)
test_df  = split_cabin_fix(test_df)

# Feature groups
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]
bool_cols = ["CryoSleep", "VIP"]
cat_cols = ["HomePlanet", "Destination", "Deck", "Side"]
drop_cols = ["Name", "Cabin"]
target_col = "Transported"

# Remove unused columns; keep target (Train only)
train_proc = train_df.drop(columns=drop_cols, errors="ignore").copy()
test_proc  = test_df.drop(columns=drop_cols, errors="ignore").copy()

# Target as {0,1} (Kaggle has True/False in 'Transported')
y_train = train_proc[target_col].astype(bool).astype(int)
X_train_in = train_proc.drop(columns=[target_col])

# --- preprocessing pipelines ---
# Numerics: median imputation + standardization (mean=0, std=1 is tanh-friendly)
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# Booleans: impute with most frequent; keep them as numeric 0/1 later
bool_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])
# Categoricals: impute most frequent + one-hot (ignore unseen categories at test time)
# Note: using sparse_output=False (sklearn>=1.2). If using an older version, use sparse=False.
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine into a single ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("bool", bool_transformer, bool_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Fit on train, transform both train and test
X_train_arr = preprocessor.fit_transform(X_train_in)
X_test_arr  = preprocessor.transform(test_proc)

# --- recover one-hot feature names for export/inspection ---
onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = []
for i, col in enumerate(cat_cols):
    for val in onehot.categories_[i]:
        cat_feature_names.append(f"{col}__{val}")

feature_names = num_cols + bool_cols + cat_feature_names

# --- convert boolean-imputed columns to {0,1} explicitly (tanh-friendly numeric) ---
b0 = len(num_cols); b1 = b0 + len(bool_cols)
X_train_arr[:, b0:b1] = (X_train_arr[:, b0:b1] > 0.5).astype(float)
X_test_arr[:, b0:b1]  = (X_test_arr[:, b0:b1] > 0.5).astype(float)

# --- export processed matrices to CSVs for downstream modeling ---
X_train_out = pd.DataFrame(X_train_arr, columns=feature_names)
X_test_out  = pd.DataFrame(X_test_arr, columns=feature_names)

X_train_out.to_csv(outdir/"X_train_processed.csv", index=False)
X_test_out.to_csv(outdir/"X_test_processed.csv", index=False)
with open(outdir/"preprocessing_feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))

# --- a few quick histograms to show the effect of scaling ---
plt.figure(figsize=(6,4))
train_df["Age"].dropna().hist(bins=30)
plt.title("Age — before scaling")
plt.tight_layout(); plt.savefig(outdir/"hist_age_before.png", dpi=150)

plt.figure(figsize=(6,4))
pd.Series(X_train_out["Age"]).hist(bins=30)
plt.title("Age — after standardization")
plt.tight_layout(); plt.savefig(outdir/"hist_age_after.png", dpi=150)

plt.figure(figsize=(6,4))
train_df["FoodCourt"].dropna().hist(bins=30)
plt.title("FoodCourt — before scaling")
plt.tight_layout(); plt.savefig(outdir/"hist_foodcourt_before.png", dpi=150)

plt.figure(figsize=(6,4))
pd.Series(X_train_out["FoodCourt"]).hist(bins=30)
plt.title("FoodCourt — after standardization")
plt.tight_layout(); plt.savefig(outdir/"hist_foodcourt_after.png", dpi=150)

print("Done. Files saved in the current folder.")
