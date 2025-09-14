#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron — Exercises 1 & 2
Author: Felipe Maluli de Carvalho Dias
Date: 13/09/2025

Requirements:
  pip install -r ../../requirements.txt

Implements a single-layer perceptron **from scratch** (NumPy only), per the activity:
- Exercise 1: linearly separable 2D Gaussians (low variance)
- Exercise 2: partially overlapping 2D Gaussians (higher variance)
- Update rule: w <- w + eta*y*x, b <- b + eta*y  (y in {-1,+1}), eta=0.01
- Stop: convergence (no updates in a full epoch) or 100 epochs, track accuracy
- Plots: data scatter, decision boundary, accuracy vs. epoch, misclassified points

Usage:
    python solution_exercises.py
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Reproducibility & outputs
# ---------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)


# ---------------------------
# Core Perceptron (NumPy only)
# ---------------------------
@dataclass
class FitLog:
    acc: List[float]
    updates_per_epoch: List[int]


class Perceptron:
    """
    Single-layer perceptron for binary classification with labels in {-1,+1}.
    Uses the classic perceptron learning rule, no regularization, full-batch loop.
    """

    def __init__(self, eta: float = 0.01, max_epochs: int = 100, seed: int = 42, shuffle: bool = False):
        self.eta = float(eta)
        self.max_epochs = int(max_epochs)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)              # allow epoch shuffling
        self.w: np.ndarray | None = None  # shape (2,)
        self.b: float | None = None

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return signed scores s = w·x + b (shape: (N,))."""
        return X @ self.w + self.b  # (N,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return labels in {-1,+1} via sign(w·x + b); zeros mapped to +1."""
        s = self.predict_raw(X)
        yhat = np.where(s >= 0.0, 1, -1)
        return yhat

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correct predictions."""
        yhat = self.predict(X)
        return float(np.mean(yhat == y))

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitLog:
        """
        Train until convergence or max_epochs.
        y must be in {-1,+1}. Weights initialized small (symmetric around 0).
        """
        rs = np.random.default_rng(self.seed)
        self.w = rs.uniform(low=-0.01, high=0.01, size=X.shape[1])
        self.b = float(rs.uniform(low=-0.01, high=0.01))
        log = FitLog(acc=[], updates_per_epoch=[])

        N = X.shape[0]
        for epoch in range(self.max_epochs):
            updates = 0

            # Shuffle order per epoch to avoid cycling on overlapping data
            if self.shuffle:
                order = rs.permutation(N)
            else:
                order = range(N)

            # one full pass; online updates for misclassified points
            # misclassified iff y * (w·x + b) <= 0
            for i in order:
                s = float(np.dot(self.w, X[i]) + self.b)
                if y[i] * s <= 0.0:
                    self.w += self.eta * y[i] * X[i]
                    self.b += self.eta * y[i]
                    updates += 1

            acc = self.accuracy(X, y)
            log.acc.append(acc)
            log.updates_per_epoch.append(updates)

            # stop if converged (no updates in a whole epoch)
            if updates == 0:
                break

        return log


# ---------------------------
# Data generation per spec
# ---------------------------
def ex1_data(n_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 1 (page 1):
      Class 0: mean=[1.5,1.5], cov=diag(0.5,0.5)
      Class 1: mean=[5,5],     cov=diag(0.5,0.5)
    Returns X in R^{2}, y in {-1,+1} with +1 for class 1.
    """
    mu0 = np.array([1.5, 1.5]); mu1 = np.array([5.0, 5.0])
    cov = np.diag([0.5, 0.5])
    X0 = rng.multivariate_normal(mu0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mu1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.full(n_per_class, -1), np.full(n_per_class, +1)])
    return X, y


def ex2_data(n_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 2 (page 2):
      Class 0: mean=[3,3], cov=diag(1.5,1.5)
      Class 1: mean=[4,4], cov=diag(1.5,1.5)
    Returns X in R^{2}, y in {-1,+1} with +1 for class 1.
    """
    mu0 = np.array([3.0, 3.0]); mu1 = np.array([4.0, 4.0])
    cov = np.diag([1.5, 1.5])
    X0 = rng.multivariate_normal(mu0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mu1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.full(n_per_class, -1), np.full(n_per_class, +1)])
    return X, y


# ---------------------------
# Plot helpers
# ---------------------------
def plot_scatter(X: np.ndarray, y: np.ndarray, title: str, out_png: str) -> None:
    plt.figure(figsize=(6, 6))
    mask0 = (y == -1); mask1 = (y == +1)
    plt.scatter(X[mask0, 0], X[mask0, 1], s=8, label="Class 0 (-1)")
    plt.scatter(X[mask1, 0], X[mask1, 1], s=8, label="Class 1 (+1)")
    plt.axis("equal"); plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()


def plot_decision_boundary(X: np.ndarray, y: np.ndarray,
                           w: np.ndarray, b: float,
                           title: str, out_png: str,
                           highlight_errors: bool = False) -> None:
    """
    Plot points and the line w·x + b = 0. Optionally highlight misclassified points.
    """
    plt.figure(figsize=(6, 6))
    mask0 = (y == -1); mask1 = (y == +1)
    plt.scatter(X[mask0, 0], X[mask0, 1], s=8, label="Class 0 (-1)")
    plt.scatter(X[mask1, 0], X[mask1, 1], s=8, label="Class 1 (+1)")

    # decision boundary line: w1*x + w2*y + b = 0 -> y = -(w1/w2)*x - b/w2
    xs = np.linspace(X[:, 0].min() - 1.0, X[:, 0].max() + 1.0, 200)
    if abs(w[1]) > 1e-12:
        ys = -(w[0] / w[1]) * xs - b / w[1]
        plt.plot(xs, ys, "k--", lw=2, label="w·x + b = 0")
    else:
        # vertical boundary (rare): x = -b/w1
        x0 = -b / w[0]
        plt.axvline(x0, color="k", linestyle="--", lw=2, label="w·x + b = 0")

    if highlight_errors:
        yhat = np.where(X @ w + b >= 0, 1, -1)
        err = (yhat != y)
        plt.scatter(X[err, 0], X[err, 1], s=25, facecolors="none",
                    edgecolors="red", label="Misclassified")

    plt.axis("equal"); plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()


def plot_curve(values: List[float], title: str, out_png: str, ylabel="Accuracy") -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(values, marker="o", ms=3)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


# ---------------------------
# Runners
# ---------------------------
def run_exercise(X: np.ndarray, y: np.ndarray, tag: str) -> None:
    # 1) Visualize data
    plot_scatter(X, y, f"{tag} — data scatter", os.path.join(ASSETS, f"{tag.lower()}_scatter.png"))

    # 2) Train perceptron per spec + Enable shuffle to avoid cycling in Exercise 2
    model = Perceptron(eta=0.01, max_epochs=100, seed=RANDOM_STATE, shuffle=True)
    log = model.fit(X, y)
    final_acc = model.accuracy(X, y)
    print(f"{tag} — epochs: {len(log.acc)} | final accuracy: {final_acc*100:.2f}% | "
          f"last-epoch updates: {log.updates_per_epoch[-1]}")

    # 3) Plots: boundary + accuracy curve (+ highlight misclassified)
    plot_decision_boundary(X, y, model.w, model.b,
                           f"{tag} — decision boundary", os.path.join(ASSETS, f"{tag.lower()}_boundary.png"),
                           highlight_errors=True)
    plot_curve(log.acc, f"{tag} — accuracy over epochs", os.path.join(ASSETS, f"{tag.lower()}_accuracy.png"))


def main():
    # Exercise 1 (linearly separable; expect quick convergence, 100% acc)
    X1, y1 = ex1_data(n_per_class=1000)
    run_exercise(X1, y1, tag="Exercise 1")

    # Exercise 2 (overlap; may not reach 100%, possible oscillations)
    X2, y2 = ex2_data(n_per_class=1000)
    run_exercise(X2, y2, tag="Exercise 2")


if __name__ == "__main__":
    main()
