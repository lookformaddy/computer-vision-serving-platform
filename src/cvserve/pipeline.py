"""Training pipeline: synthetic corpus -> features -> standardizer ->
ensemble + histogram fallback -> offline evaluation -> ModelVersion.

``train_version`` is the single entry point for building a servable
version: it trains the ensemble (logistic regression + k-NN) and the
histogram-only fallback, evaluates on held-out data (accuracy, macro
F1, per-class F1, confusion matrix, ECE, latency percentiles), and
computes the drift reference (per-feature decile bin edges, per-class
centroids, OOD threshold at the 99th percentile of training distances
to the true class centroid).
"""
from __future__ import annotations

import time

from cvserve.evaluation import (
    evaluate, expected_calibration_error, latency_percentiles,
    majority_baseline_accuracy,
)
from cvserve.features import (
    HIST_SLICE, FeatureStandardizer, extract_features,
)
from cvserve.images import CLASSES, generate_dataset
from cvserve.models import (
    EnsembleClassifier, HistogramFallback, KNNClassifier,
    LogisticRegression,
)
from cvserve.registry import ModelVersion

__all__ = ["train_version"]


def _decile_edges(values: list[float]) -> list[float]:
    ordered = sorted(values)
    n = len(ordered)
    edges: list[float] = []
    for k in range(1, 10):
        idx = min(n - 1, int(k * n / 10))
        edges.append(ordered[idx])
    # De-duplicate while keeping order (constant features collapse).
    deduped: list[float] = []
    for e in edges:
        if not deduped or e > deduped[-1]:
            deduped.append(e)
    return deduped


def train_version(version: str, seed: int = 42, n_per_class: int = 100,
                  epochs: int = 120, learning_rate: float = 0.5,
                  knn_k: int = 5, logreg_weight: float = 0.7
                  ) -> tuple[ModelVersion, dict]:
    train, test = generate_dataset(n_per_class=n_per_class, seed=seed)
    X_train = [extract_features(ex.image) for ex in train]
    y_train = [ex.label for ex in train]
    X_test = [extract_features(ex.image) for ex in test]
    y_test = [ex.label for ex in test]

    standardizer = FeatureStandardizer().fit(X_train)
    Xs_train = standardizer.transform_many(X_train)
    Xs_test = standardizer.transform_many(X_test)

    logreg = LogisticRegression(epochs=epochs, seed=seed,
                                learning_rate=learning_rate)
    logreg.fit(Xs_train, y_train)
    knn = KNNClassifier(k=knn_k).fit(Xs_train, y_train)
    ensemble = EnsembleClassifier(logreg=logreg, knn=knn,
                                  logreg_weight=logreg_weight)

    hist_idx = list(range(HIST_SLICE.start, HIST_SLICE.stop))
    hist_model = LogisticRegression(epochs=epochs, seed=seed,
                                    learning_rate=learning_rate)
    hist_model.fit([[v[i] for i in hist_idx] for v in Xs_train], y_train)
    fallback = HistogramFallback(model=hist_model, indices=hist_idx)

    # Offline evaluation of the ensemble on held-out data.
    probs_list: list[dict[str, float]] = []
    latencies: list[float] = []
    for vec in Xs_test:
        t0 = time.perf_counter()
        probs_list.append(ensemble.predict_proba(vec))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    metrics = evaluate(probs_list, y_test)
    ece = expected_calibration_error(probs_list, y_test)
    latency = latency_percentiles(latencies)

    # Baselines the gates must beat.
    fallback_probs = [fallback.predict_proba(vec) for vec in Xs_test]
    fallback_metrics = evaluate(fallback_probs, y_test)
    majority_acc = majority_baseline_accuracy(y_test)

    # Drift reference from the training set.
    ref_bin_edges = [
        _decile_edges([row[j] for row in Xs_train])
        for j in range(len(Xs_train[0]))
    ]
    centroids: dict[str, list[float]] = {}
    for label in CLASSES:
        rows = [vec for vec, y in zip(Xs_train, y_train) if y == label]
        if rows:
            centroids[label] = [
                sum(row[j] for row in rows) / len(rows)
                for j in range(len(rows[0]))
            ]
    train_dists: list[float] = []
    for vec, y in zip(Xs_train, y_train):
        centroid = centroids[y]
        train_dists.append(sum((a - b) ** 2 for a, b in zip(vec, centroid))
                           ** 0.5)
    train_dists.sort()
    ood_threshold = train_dists[min(len(train_dists) - 1,
                                    int(0.99 * len(train_dists)))]

    model_version = ModelVersion(
        version=version, labels=sorted(set(y_train)),
        standardizer=standardizer, classifier=ensemble, fallback=fallback,
        metrics=metrics, p95_latency_ms=latency["p95"], ece=ece,
        baseline_accuracy=fallback_metrics.accuracy,
        majority_accuracy=majority_acc,
        n_train=len(train), n_test=len(test), seed=seed,
        ref_bin_edges=ref_bin_edges, centroids=centroids,
        ood_threshold=ood_threshold,
        notes=f"synthetic {len(CLASSES)}-class corpus, "
              f"ensemble logreg+kNN({knn_k})",
    )
    report = {
        "version": version,
        "n_train": len(train),
        "n_test": len(test),
        "vocab_size": None,
        "n_features": len(X_train[0]),
        "final_loss": (logreg.loss_history_[-1]
                       if logreg.loss_history_ else 0.0),
        "metrics": metrics,
        "ece": ece,
        "latency": latency,
        "baseline_accuracy": fallback_metrics.accuracy,
        "majority_accuracy": majority_acc,
        "ood_threshold": ood_threshold,
    }
    return model_version, report
