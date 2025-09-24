# -*- coding: utf-8 -*-
"""
Activity — MLP
Author: Felipe Maluli de Carvalho Dias
Date: 24/09/2025

Requirements:
  pip install -r ../../requirements.txt

This script trains and evaluates a **hand‑implemented** Multi‑Layer Perceptron
using only **NumPy** for array math. It follows the activity's rules:

- Exercises **2–4**: datasets are generated with `sklearn.datasets.make_classification`.
- The **MLP itself** (forward pass, activations, losses, gradients, GD updates) is
  implemented **from scratch** (no TensorFlow/PyTorch/`sklearn` estimators).
- Results (loss curves, confusion matrix) are saved into `./assets` for the report.

Usage
-----
    python solution_exercises.py

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Literal

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


@dataclass
class TrainConfig:
    """Hyperparameters and reproducibility settings used across exercises."""

    seed: int
    epochs: int
    lr: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> np.random.RandomState:
    """Create a deterministic RandomState and set NumPy global seed too."""
    np.random.seed(seed)
    return np.random.RandomState(seed)


def shuffle_split(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple 80/20 split using a deterministic permutation (no sklearn split)."""
    rs = seed_everything(seed)
    idx = rs.permutation(len(X))
    split = int(len(X) * (1.0 - test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels (N,) to one‑hot array (N, C)."""
    y = y.reshape(-1)
    M = np.zeros((len(y), num_classes))
    M[np.arange(len(y)), y] = 1.0
    return M


def plot_curve(values: List[float], title: str, out_png: str) -> None:
    """Save a simple training‑loss curve."""
    plt.figure()
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(out_png)
    plt.close()


# ---------------------------------------------------------------------------
# Core MLP (NumPy only)
# ---------------------------------------------------------------------------

class MLP:
    """
    Minimal MLP with tanh hidden layers and either
    - 'binary' output: sigmoid + binary cross‑entropy
    - 'multiclass' output: softmax + categorical cross‑entropy

    The entire forward pass, gradient derivations, and parameter updates are
    implemented from scratch to comply with the activity rules.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        task: Literal["binary", "multiclass"],
        seed: int = 42,
    ) -> None:
        assert len(layer_sizes) >= 2, "Need at least input and output layer sizes."
        self.layer_sizes = layer_sizes
        self.task = task
        rs = seed_everything(seed)

        # Xavier/Glorot uniform initialization
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(rs.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    # ---- Activation functions (and their derivatives) ---------------------
    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def dtanh(z: np.ndarray) -> np.ndarray:
        t = np.tanh(z)
        return 1.0 - t * t

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        zmax = np.max(z, axis=1, keepdims=True)
        e = np.exp(z - zmax)
        return e / np.sum(e, axis=1, keepdims=True)

    # ---- Loss functions ----------------------------------------------------
    @staticmethod
    def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
        """Binary cross‑entropy averaged over the batch."""
        p = np.clip(p, eps, 1 - eps)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    @staticmethod
    def cce(y_onehot: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
        """Categorical cross‑entropy averaged over the batch."""
        p = np.clip(p, eps, 1 - eps)
        return float(-np.sum(y_onehot * np.log(p)) / y_onehot.shape[0])

    # ---- Forward / Backward / Step ----------------------------------------
    def forward(self, X: np.ndarray, cache: bool = True):
        """
        Forward pass.
        Returns either (yhat, caches) for training or just yhat for inference.
        """
        a = X
        caches = []  # (a_prev, z, h) per layer; last item stores (a_prev, z_L, yhat)
        # Hidden layers (tanh)
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            h = self.tanh(z)
            caches.append((a, z, h))
            a = h
        # Output layer
        zL = a @ self.W[-1] + self.b[-1]
        yhat = self.sigmoid(zL) if self.task == "binary" else self.softmax(zL)
        caches.append((a, zL, yhat))
        return (yhat, caches) if cache else yhat

    def backward(self, caches, y: np.ndarray):
        """Backpropagate the loss through the network to compute gradients."""
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        a_prev, zL, yhat = caches[-1]
        N = yhat.shape[0]
        # For sigmoid+BCE and softmax+CCE the derivative w.r.t. zL simplifies to (yhat - y)/N
        dZ = (yhat - y) / N

        # Last layer grads
        hidden_act = caches[-2][2] if len(caches) > 1 else caches[-1][0]
        grads_W[-1] = hidden_act.T @ dZ
        grads_b[-1] = np.sum(dZ, axis=0)

        # Propagate to previous layer
        dA = dZ @ self.W[-1].T

        # Hidden layers (all tanh)
        for l in range(len(self.W) - 2, -1, -1):
            a_prev, z, h = caches[l]
            dZ = dA * self.dtanh(z)
            grads_W[l] = a_prev.T @ dZ
            grads_b[l] = np.sum(dZ, axis=0)
            if l > 0:
                dA = dZ @ self.W[l].T

        return grads_W, grads_b

    def step(self, grads_W, grads_b, lr: float) -> None:
        """Gradient‑descent update for all parameters."""
        for i in range(len(self.W)):
            self.W[i] -= lr * grads_W[i]
            self.b[i] -= lr * grads_b[i]

    def fit(self, X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> dict:
        """Train with full‑batch gradient descent."""
        history = {"loss": []}
        for ep in range(cfg.epochs):
            yhat, caches = self.forward(X, cache=True)
            loss = self.bce(y, yhat) if self.task == "binary" else self.cce(y, yhat)
            gW, gB = self.backward(caches, y)
            self.step(gW, gB, cfg.lr)
            history["loss"].append(loss)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat = self.forward(X, cache=False)
        if self.task == "binary":
            return (yhat >= 0.5).astype(int)
        return np.argmax(yhat, axis=1)


# ---------------------------------------------------------------------------
# Data generation (make_classification) — as required by the assignment
# ---------------------------------------------------------------------------

def make_ex2_binary(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 2 dataset: 2 features, 2 classes, with asymmetry:
      - class 0 has 1 cluster
      - class 1 has 2 clusters
    We create the two class subsets separately and then combine them.
    """
    n0 = n_samples // 2
    n1 = n_samples - n0

    X0, y0 = make_classification(
        n_samples=int(n0 * 2),
        n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
        n_classes=2, n_clusters_per_class=1,
        weights=[0.95, 0.05], class_sep=1.5, flip_y=0.0,
        random_state=seed,
    )
    X1, y1 = make_classification(
        n_samples=int(n1 * 2),
        n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
        n_classes=2, n_clusters_per_class=2,
        weights=[0.05, 0.95], class_sep=1.5, flip_y=0.0,
        random_state=seed + 1,
    )
    X0 = X0[y0 == 0][:n0]
    X1 = X1[y1 == 1][:n1]

    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1)).reshape(-1, 1)

    # final shuffle for reproducibility
    rs = seed_everything(seed + 2)
    p = rs.permutation(len(X))
    return X[p], y[p]


def make_ex3_multiclass(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 3 dataset: 4 features, 3 classes, per-class clusters [2, 3, 4].
    We generate each class subset separately and concatenate.
    """
    assert n_samples % 3 == 0, "Use a multiple of 3 for equal class counts."
    per = n_samples // 3

    def gen(target: int, clusters: int, offset: int):
        Xc, yc = make_classification(
            n_samples=per * 3,  # oversample then filter to the desired class
            n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
            n_classes=3, n_clusters_per_class=clusters,
            weights=[0.9 if i == target else 0.05 for i in range(3)],
            class_sep=1.8, flip_y=0.0, random_state=seed + offset,
        )
        return Xc[yc == target][:per], np.full(per, target, dtype=int)

    X0, y0 = gen(0, 2, 0)
    X1, y1 = gen(1, 3, 1)
    X2, y2 = gen(2, 4, 2)
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])

    rs = seed_everything(seed + 9)
    p = rs.permutation(len(X))
    return X[p], y[p]


# ---------------------------------------------------------------------------
# Exercise runners
# ---------------------------------------------------------------------------

def run_ex2() -> float:
    X, y = make_ex2_binary(n_samples=1000, seed=7)
    Xtr, Xte, ytr, yte = shuffle_split(X, y, test_size=0.2, seed=7)

    model = MLP([2, 10, 1], task="binary", seed=7)
    cfg = TrainConfig(seed=7, epochs=300, lr=0.08)
    hist = model.fit(Xtr, ytr, cfg)

    # Eval
    pred = model.predict(Xte).reshape(-1)
    acc = float((pred == yte.reshape(-1)).mean())

    plot_curve(hist["loss"], "Exercise 2 — Training Loss", os.path.join(ASSETS_DIR, "ex2_loss.png"))
    print(f"Exercise 2 — test accuracy: {acc*100:.2f}%")
    return acc


def run_ex3() -> float:
    X, y = make_ex3_multiclass(n_samples=1500, seed=9)
    Xtr, Xte, ytr, yte = shuffle_split(X, y.reshape(-1, 1), test_size=0.2, seed=9)

    C = int(np.unique(y).size)
    model = MLP([4, 16, C], task="multiclass", seed=9)  # exact same class as Ex2
    cfg = TrainConfig(seed=9, epochs=350, lr=0.07)
    ytr_oh = one_hot(ytr.reshape(-1), C)
    hist = model.fit(Xtr, ytr_oh, cfg)

    pred = model.predict(Xte)
    acc = float((pred.reshape(-1) == yte.reshape(-1)).mean())

    plot_curve(hist["loss"], "Exercise 3 — Training Loss", os.path.join(ASSETS_DIR, "ex3_loss.png"))

    # Confusion matrix (simple counts)
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(yte.reshape(-1), pred.reshape(-1)):
        cm[int(t), int(p)] += 1

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Exercise 3 — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(C):
        for j in range(C):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(os.path.join(ASSETS_DIR, "ex3_confusion.png"))
    plt.close()

    print(f"Exercise 3 — test accuracy: {acc*100:.2f}%")
    return acc


def run_ex4() -> float:
    # Reuse Exercise 3's dataset/split for a fair comparison
    X, y = make_ex3_multiclass(n_samples=1500, seed=9)
    Xtr, Xte, ytr, yte = shuffle_split(X, y.reshape(-1, 1), test_size=0.2, seed=9)

    C = int(np.unique(y).size)
    model = MLP([4, 24, 12, C], task="multiclass", seed=10)  # deeper: 2 hidden layers
    cfg = TrainConfig(seed=10, epochs=350, lr=0.07)
    ytr_oh = one_hot(ytr.reshape(-1), C)
    hist = model.fit(Xtr, ytr_oh, cfg)

    pred = model.predict(Xte)
    acc = float((pred.reshape(-1) == yte.reshape(-1)).mean())

    plot_curve(hist["loss"], "Exercise 4 — Training Loss (Deeper MLP)",
               os.path.join(ASSETS_DIR, "ex4_loss.png"))
    print(f"Exercise 4 — test accuracy: {acc*100:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    acc2 = run_ex2()
    acc3 = run_ex3()
    acc4 = run_ex4()

    # Optional summary line for copy/paste into the report
    print(f"\nSummary — Ex2: {acc2*100:.2f}% | Ex3: {acc3*100:.2f}% | Ex4: {acc4*100:.2f}%")


if __name__ == "__main__":
    main()
