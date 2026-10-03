"""
learned_policy.py — Learned Page Replacement Using a Decision Tree

Overview
========
A DecisionTreeClassifier (scikit-learn) is trained to predict which page
in memory should be evicted next.

Training
--------
    Training data is collected from Phase 1 of the trace only (the
    locality-heavy phase).  Labels are generated using Belady's optimal
    algorithm, which has access to future page accesses.

    IMPORTANT: Future information is used ONLY to generate labels.
    The three input features (recency, frequency, recency_ratio) are
    strictly causal — computed only from history up to the current step.

Inference
---------
    During page replacement simulation, when an eviction is required:
        1. Compute features for every candidate page in memory.
        2. Ask the model for the probability that each page is the optimal
           eviction candidate (predict_proba).
        3. Evict the page with the HIGHEST predicted probability of being
           the best eviction candidate (class 1).
        4. Tie-break: if all probabilities are equal, evict the page with
           the greatest recency (least recently used fallback).

The model is intentionally shallow (max_depth=4) to avoid overfitting
and to remain interpretable.

Model metrics are logged for honesty:
    accuracy, precision, recall, F1 (on held-out evaluation data).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from src.features import compute_candidate_features


FEATURE_NAMES = ["recency", "frequency", "recency_ratio"]


def train_model(X, y, random_state=42, test_size=0.25):
    """
    Train a DecisionTreeClassifier on the provided samples and return
    both the fitted model and evaluation metrics.

    Parameters
    ----------
    X : list[list[float]]   — feature vectors
    y : list[int]           — binary labels (1 = optimal eviction target)
    random_state : int
    test_size : float       — fraction of data used for evaluation

    Returns
    -------
    model   : fitted DecisionTreeClassifier
    metrics : dict with accuracy, precision, recall, f1, n_train, n_test
    report  : str — full sklearn classification_report
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    # Shallow tree: readable, less prone to overfitting
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "n_train":   len(y_train),
        "n_test":    len(y_test),
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    report = classification_report(y_test, y_pred, zero_division=0)

    return model, metrics, report


def run_learned_policy(trace, num_frames, model):
    """
    Simulate page replacement using the trained Decision Tree model.

    At each eviction point:
        - Compute features for every page currently in memory.
        - Use model.predict_proba to score each candidate.
        - Evict the candidate with the highest predicted probability
          of being the optimal eviction target (class 1).

    Parameters
    ----------
    trace      : list[int]
    num_frames : int
    model      : fitted DecisionTreeClassifier

    Returns
    -------
    dict with keys:
        total_accesses, hits, faults, hit_ratio, fault_ratio, per_access
    """
    frames = set()
    access_history = {}   # page → list of past access indices

    hits = 0
    faults = 0
    per_access = []

    for idx, page in enumerate(trace):
        if page in frames:
            # Hit
            access_history.setdefault(page, []).append(idx)
            hits += 1
            per_access.append({
                "page": page, "decision": "hit",
                "evicted": None, "frames_snapshot": frozenset(frames),
            })
        else:
            # Fault
            faults += 1
            evicted = None
            confidence = 0.0

            if len(frames) == num_frames:
                # Compute features for all candidates
                candidate_features = compute_candidate_features(
                    frames, idx, access_history
                )

                # Build feature matrix for batch prediction
                pages_list = [cf["page"] for cf in candidate_features]
                feat_matrix = np.array(
                    [[cf["recency"], cf["frequency"], cf["recency_ratio"]]
                     for cf in candidate_features],
                    dtype=float,
                )

                # Predict probability of being the optimal eviction target
                if feat_matrix.shape[0] > 0:
                    # predict_proba returns [[prob_0, prob_1], ...]
                    proba = model.predict_proba(feat_matrix)
                    # Column 1 = probability of class 1 (= should evict)
                    # Handle edge case where model only saw one class in training
                    if proba.shape[1] == 2:
                        evict_proba = proba[:, 1]
                    else:
                        # Fallback: use recency as tie-breaker
                        evict_proba = np.array(
                            [cf["recency"] for cf in candidate_features],
                            dtype=float,
                        )

                    best_idx = int(np.argmax(evict_proba))
                    evicted = pages_list[best_idx]
                    confidence = float(evict_proba[best_idx])
                else:
                    # Fallback: evict page with greatest recency
                    evicted = max(
                        frames,
                        key=lambda p: idx - (access_history[p][-1]
                                             if access_history.get(p) else -1),
                    )
                    confidence = 0.50

                frames.discard(evicted)

            frames.add(page)
            access_history.setdefault(page, []).append(idx)

            per_access.append({
                "page": page, "decision": "fault",
                "evicted": evicted, "frames_snapshot": frozenset(frames),
                "index": idx,
                "confidence": confidence,
            })

    total = len(trace)
    return {
        "total_accesses": total,
        "hits":           hits,
        "faults":         faults,
        "hit_ratio":      hits / total if total > 0 else 0.0,
        "fault_ratio":    faults / total if total > 0 else 0.0,
        "per_access":     per_access,
    }
