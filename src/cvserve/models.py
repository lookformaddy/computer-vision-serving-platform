"""From-scratch classifiers on dense feature vectors (pure stdlib).

- :class:`LogisticRegression`: multinomial logistic regression, softmax
  + cross-entropy, full-batch gradient descent with L2 and learning-rate
  decay. The serving model for the full 45-feature vector.
- :class:`KNNClassifier`: k-nearest neighbours with inverse-distance
  weighted voting over the standardized feature space.
- :class:`EnsembleClassifier`: probability-averaged blend of the two
  (default 0.7 logistic regression / 0.3 k-NN) — the champion model.
- :class:`HistogramFallback`: a logistic regression trained on the
  8-bin intensity histogram slice only. Deliberately weaker, but tiny
  and fast — it is the latency-budget fallback, not a competitor.

``predict_proba`` returns ``{label: probability}``; ``explain`` returns
the top contributing features for a prediction as ``(name, contribution)``
pairs where contribution is ``standardized_feature * weight`` for the
predicted class — faithful by construction for the linear model.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = [
    "LogisticRegression", "KNNClassifier", "EnsembleClassifier",
    "HistogramFallback",
]


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class LogisticRegression:
    labels: list[str] = field(default_factory=list)
    learning_rate: float = 0.5
    epochs: int = 100
    l2: float = 1e-4
    lr_decay: float = 0.98
    seed: int = 42

    weights_: list[list[float]] = field(default_factory=list, init=False)
    bias_: list[float] = field(default_factory=list, init=False)
    n_features_: int = field(default=0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: list[list[float]], y: list[str]
            ) -> "LogisticRegression":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not X:
            raise ValueError("cannot fit on empty data")
        self.labels = sorted(set(y))
        label_to_idx = {lab: i for i, lab in enumerate(self.labels)}
        n_classes = len(self.labels)
        self.n_features_ = len(X[0])
        rng = random.Random(self.seed)
        self.weights_ = [
            [rng.gauss(0.0, 0.01) for _ in range(self.n_features_)]
            for _ in range(n_classes)
        ]
        self.bias_ = [0.0] * n_classes
        lr = self.learning_rate
        n = len(X)
        for _epoch in range(self.epochs):
            grad_w = [[0.0] * self.n_features_ for _ in range(n_classes)]
            grad_b = [0.0] * n_classes
            loss = 0.0
            for vec, label in zip(X, y):
                scores = [
                    self.bias_[c]
                    + sum(w * v for w, v in zip(self.weights_[c], vec))
                    for c in range(n_classes)
                ]
                probs = _softmax(scores)
                true_idx = label_to_idx[label]
                loss -= math.log(max(probs[true_idx], 1e-12))
                for c in range(n_classes):
                    err = probs[c] - (1.0 if c == true_idx else 0.0)
                    if err == 0.0:
                        continue
                    grad_b[c] += err
                    gw = grad_w[c]
                    for j, v in enumerate(vec):
                        gw[j] += err * v
            for c in range(n_classes):
                w = self.weights_[c]
                gw = grad_w[c]
                for j in range(self.n_features_):
                    w[j] -= lr * (gw[j] / n + self.l2 * w[j])
                self.bias_[c] -= lr * grad_b[c] / n
            self.loss_history_.append(loss / n)
            lr *= self.lr_decay
        return self

    def _scores(self, vec: list[float]) -> list[float]:
        return [
            self.bias_[c]
            + sum(w * v for w, v in zip(self.weights_[c], vec))
            for c in range(len(self.labels))
        ]

    def predict_proba(self, vec: list[float]) -> dict[str, float]:
        if not self.labels:
            raise ValueError("classifier is not fitted")
        probs = _softmax(self._scores(vec))
        return {lab: p for lab, p in zip(self.labels, probs)}

    def predict(self, vec: list[float]) -> str:
        probs = self.predict_proba(vec)
        return max(probs, key=lambda lab: probs[lab])

    def explain(self, vec: list[float], feature_names: list[str],
                top_k: int = 5) -> list[tuple[str, float]]:
        if not self.labels:
            raise ValueError("classifier is not fitted")
        pred = self.predict(vec)
        c = self.labels.index(pred)
        contribs: list[tuple[str, float]] = []
        for j, v in enumerate(vec):
            w = self.weights_[c][j]
            if w == 0.0 or v == 0.0:
                continue
            name = (feature_names[j] if j < len(feature_names)
                    else f"f{j}")
            contribs.append((name, v * w))
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return contribs[:top_k]


@dataclass
class KNNClassifier:
    k: int = 5

    X_: list[list[float]] = field(default_factory=list, init=False)
    y_: list[str] = field(default_factory=list, init=False)
    labels: list[str] = field(default_factory=list, init=False)

    def fit(self, X: list[list[float]], y: list[str]) -> "KNNClassifier":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not X:
            raise ValueError("cannot fit on empty data")
        self.X_ = [list(row) for row in X]
        self.y_ = list(y)
        self.labels = sorted(set(y))
        return self

    def predict_proba(self, vec: list[float]) -> dict[str, float]:
        if not self.X_:
            raise ValueError("classifier is not fitted")
        dists = [
            (sum((a - b) ** 2 for a, b in zip(vec, row)), label)
            for row, label in zip(self.X_, self.y_)
        ]
        dists.sort(key=lambda t: t[0])
        scores = {lab: 0.0 for lab in self.labels}
        for dist, label in dists[:self.k]:
            scores[label] += 1.0 / (math.sqrt(dist) + 1e-6)
        total = sum(scores.values()) or 1.0
        return {lab: s / total for lab, s in scores.items()}

    def predict(self, vec: list[float]) -> str:
        probs = self.predict_proba(vec)
        return max(probs, key=lambda lab: probs[lab])


@dataclass
class EnsembleClassifier:
    """Probability-averaged blend of logistic regression and k-NN."""

    logreg: LogisticRegression
    knn: KNNClassifier
    logreg_weight: float = 0.7

    @property
    def labels(self) -> list[str]:
        return self.logreg.labels

    def predict_proba(self, vec: list[float]) -> dict[str, float]:
        p_lr = self.logreg.predict_proba(vec)
        p_knn = self.knn.predict_proba(vec)
        w = self.logreg_weight
        return {lab: w * p_lr.get(lab, 0.0) + (1 - w) * p_knn.get(lab, 0.0)
                for lab in self.labels}

    def predict(self, vec: list[float]) -> str:
        probs = self.predict_proba(vec)
        return max(probs, key=lambda lab: probs[lab])

    def explain(self, vec: list[float], feature_names: list[str],
                top_k: int = 5) -> list[tuple[str, float]]:
        # Explanations come from the linear member (faithful by
        # construction); the k-NN member has no per-feature weights.
        return self.logreg.explain(vec, feature_names, top_k=top_k)


@dataclass
class HistogramFallback:
    """Lightweight fallback: logistic regression on the 8-bin intensity
    histogram slice of the feature vector only. Fast and always
    available; deliberately weaker than the full ensemble.
    """

    model: LogisticRegression
    indices: list[int]

    def predict_proba(self, vec: list[float]) -> dict[str, float]:
        return self.model.predict_proba([vec[i] for i in self.indices])

    def predict(self, vec: list[float]) -> str:
        probs = self.predict_proba(vec)
        return max(probs, key=lambda lab: probs[lab])
