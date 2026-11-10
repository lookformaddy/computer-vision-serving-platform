"""Serving layer: canary + shadow deployment, deterministic hash A/B
routing, latency budgets with a histogram-only fallback, and drift
monitoring (feature-distribution PSI over a rolling window, OOD
detection via distance-to-centroid, per-request z-score outliers).

Request flow for ``predict(image, request_id)``:
1. Features are extracted and standardized; the drift monitor scores
   the input (OOD distance to the nearest class centroid, fraction of
   features with |z| > 3) and adds it to the rolling window.
2. A deterministic hash of ``request_id`` decides the serving arm:
   ``hash % 100 < canary_pct`` routes to the challenger (canary), the
   rest to the champion. No request_id -> champion.
3. The chosen model predicts under a latency budget. If it exceeds the
   budget (or raises), the fast histogram-only fallback answers instead
   and the outcome is recorded as a fallback.
4. If a challenger exists and the request went to the champion, the
   challenger is also scored in shadow mode; champion/challenger
   agreement is recorded but the shadow result is never returned.
5. The response carries the prediction, confidence, top contributing
   features, which arm served it, latency, and drift flags.

``predict_batch`` scores a list of images through the same path (the
batching entry point for offline and bulk callers).
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field

from cvserve.features import FEATURE_NAMES, extract_features
from cvserve.models import EnsembleClassifier
from cvserve.registry import ModelRegistry, ModelVersion

__all__ = ["ServingConfig", "Prediction", "DriftMonitor", "ServingLayer",
           "hash_arm"]

PSI_ALERT_THRESHOLD = 0.20
OOD_RATE_ALERT_THRESHOLD = 0.10
Z_OUTLIER_ALERT_THRESHOLD = 0.20
PSI_EPS = 1e-4


def hash_arm(request_id: str) -> int:
    """Deterministic 0..99 bucket for a request id (stable across runs)."""
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


@dataclass
class ServingConfig:
    canary_pct: int = 10            # % of hashed traffic to the challenger
    latency_budget_ms: float = 100.0  # model path budget; over -> fallback
    psi_alert_threshold: float = PSI_ALERT_THRESHOLD
    shadow_enabled: bool = True
    drift_window: int = 200         # rolling window for PSI


@dataclass
class Prediction:
    label: str
    confidence: float
    explanation: list[tuple[str, float]] = field(default_factory=list)
    served_by: str = "champion"     # champion | challenger | fallback
    model_version: str | None = None
    latency_ms: float = 0.0
    fallback_reason: str | None = None
    ood_distance: float = 0.0
    ood_flag: bool = False
    z_outlier_rate: float = 0.0
    drift_flag: bool = False


@dataclass
class DriftMonitor:
    """Rolling-window drift against the champion's training reference.

    Per-request signals: distance to the nearest class centroid in
    standardized feature space (OOD) and the fraction of features with
    |z| > 3. Aggregated signal: per-feature PSI of the rolling window
    against the training decile reference (PSI > 0.2 is the standard
    "significant shift" threshold).
    """
    psi_alert_threshold: float = PSI_ALERT_THRESHOLD
    window_size: int = 200
    window: deque = field(default_factory=lambda: deque(maxlen=200))
    n_requests: int = 0
    ood_flagged: int = 0
    total_ood_distance: float = 0.0
    alerts: list[dict] = field(default_factory=list)

    def _psi(self, ref_bin_edges: list[list[float]]) -> dict[str, float]:
        if len(self.window) < 30 or not ref_bin_edges:
            return {}
        n = len(self.window)
        n_features = len(ref_bin_edges)
        out: dict[str, float] = {}
        for j in range(n_features):
            edges = ref_bin_edges[j]
            counts = [0] * (len(edges) + 1)
            for vec in self.window:
                v = vec[j]
                b = sum(1 for e in edges if v > e)
                counts[b] += 1
            psi = 0.0
            for count in counts:
                actual = max(count / n, PSI_EPS)
                expected = max(1.0 / len(counts), PSI_EPS)
                psi += (actual - expected) * math.log(actual / expected)
            out[FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"f{j}"] = psi
        return out

    def score(self, vec: list[float], version: ModelVersion
              ) -> tuple[float, bool, float, bool]:
        """Score one standardized feature vector.

        Returns (ood_distance, ood_flag, z_outlier_rate, drift_flag).
        """
        self.n_requests += 1
        ood_dist = min(
            (math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, centroid)))
             for centroid in version.centroids.values()),
            default=0.0)
        self.total_ood_distance += ood_dist
        ood_flag = ood_dist > version.ood_threshold
        self.ood_flagged += int(ood_flag)
        z_outliers = sum(1 for v in vec if abs(v) > 3.0) / len(vec) \
            if vec else 0.0
        self.window.append(list(vec))
        drift_flag = ood_flag or z_outliers > Z_OUTLIER_ALERT_THRESHOLD
        return ood_dist, ood_flag, z_outliers, drift_flag

    def psi_summary(self, ref_bin_edges: list[list[float]]) -> dict:
        psi = self._psi(ref_bin_edges)
        if not psi:
            return {"window_n": len(self.window), "psi_max": 0.0,
                    "psi_by_feature": {}}
        worst = max(psi.items(), key=lambda kv: kv[1])
        return {"window_n": len(self.window),
                "psi_max": round(worst[1], 4),
                "psi_worst_feature": worst[0],
                "psi_by_feature": {k: round(v, 4) for k, v in psi.items()}}

    def check_alerts(self, ref_bin_edges: list[list[float]]) -> None:
        """Record an alert when window PSI or the OOD rate crosses its
        threshold. Checked when the window fills in blocks of 50."""
        if len(self.window) < 50 or len(self.window) % 50 != 0:
            return
        psi = self._psi(ref_bin_edges)
        if psi:
            worst_feature, worst = max(psi.items(), key=lambda kv: kv[1])
            if worst > self.psi_alert_threshold:
                self.alerts.append({
                    "kind": "psi", "feature": worst_feature,
                    "value": round(worst, 4),
                    "threshold": self.psi_alert_threshold,
                    "n_requests": self.n_requests,
                })
        ood_rate = self.ood_flagged / self.n_requests if self.n_requests else 0.0
        if ood_rate > OOD_RATE_ALERT_THRESHOLD:
            self.alerts.append({
                "kind": "ood_rate", "value": round(ood_rate, 4),
                "threshold": OOD_RATE_ALERT_THRESHOLD,
                "n_requests": self.n_requests,
            })

    def summary(self, ref_bin_edges: list[list[float]]) -> dict:
        mean_ood = (self.total_ood_distance / self.n_requests
                    if self.n_requests else 0.0)
        return {
            "n_requests": self.n_requests,
            "ood_flagged": self.ood_flagged,
            "mean_ood_distance": round(mean_ood, 4),
            "alerts": len(self.alerts),
            **self.psi_summary(ref_bin_edges),
        }


class ServingLayer:
    def __init__(self, registry: ModelRegistry,
                 config: ServingConfig | None = None):
        self.registry = registry
        self.config = config or ServingConfig()
        self.drift = DriftMonitor(
            psi_alert_threshold=self.config.psi_alert_threshold,
            window_size=self.config.drift_window)
        self.drift.window = deque(maxlen=self.config.drift_window)
        self.shadow_agreements: list[bool] = []
        self.fallbacks: int = 0
        self.served: dict[str, int] = {"champion": 0, "challenger": 0,
                                       "fallback": 0}

    # -- model scoring ---------------------------------------------------
    @staticmethod
    def _model_predict(version: ModelVersion,
                       image: list[list[float]]) -> Prediction:
        vec = version.standardizer.transform(extract_features(image))
        probs = version.classifier.predict_proba(vec)
        label = max(probs, key=lambda lab: probs[lab])
        explanation = version.classifier.explain(
            vec, FEATURE_NAMES)
        return Prediction(label=label, confidence=probs[label],
                          explanation=explanation,
                          model_version=version.version)

    def _fallback_predict(self, version: ModelVersion,
                          image: list[list[float]],
                          reason: str) -> Prediction:
        vec = version.standardizer.transform(extract_features(image))
        probs = version.fallback.predict_proba(vec)
        label = max(probs, key=lambda lab: probs[lab])
        return Prediction(label=label, confidence=probs[label],
                          served_by="fallback", model_version=version.version,
                          fallback_reason=reason)

    # -- public API ------------------------------------------------------
    def predict(self, image: list[list[float]],
                request_id: str | None = None) -> Prediction:
        champion = self.registry.get_champion()
        if champion is None:
            raise ValueError("no champion registered")
        started = time.perf_counter()

        # 1. Drift scoring against the champion's training reference.
        vec = champion.standardizer.transform(extract_features(image))
        ood_dist, ood_flag, z_rate, drift_flag = self.drift.score(
            vec, champion)
        self.drift.check_alerts(champion.ref_bin_edges)

        # 2. Deterministic hash A/B routing.
        challenger = self.registry.get_challenger()
        arm = "champion"
        version = champion
        if (challenger is not None and request_id is not None
                and hash_arm(request_id) < self.config.canary_pct):
            arm = "challenger"
            version = challenger

        # 3. Predict under the latency budget, else fall back.
        prediction: Prediction | None = None
        try:
            prediction = self._model_predict(version, image)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if latency_ms > self.config.latency_budget_ms:
                raise TimeoutError(
                    f"latency {latency_ms:.1f}ms exceeded budget "
                    f"{self.config.latency_budget_ms}ms")
        except Exception as exc:  # noqa: BLE001 - fallback is the point
            latency_ms = (time.perf_counter() - started) * 1000.0
            prediction = self._fallback_predict(
                champion, image,
                reason=f"{type(exc).__name__}: {exc}")
            prediction.latency_ms = latency_ms
            self.fallbacks += 1
            self.served["fallback"] += 1
            prediction.ood_distance = ood_dist
            prediction.ood_flag = ood_flag
            prediction.z_outlier_rate = z_rate
            prediction.drift_flag = drift_flag
            return prediction

        # 4. Shadow-score the challenger on champion traffic.
        if (self.config.shadow_enabled and challenger is not None
                and arm == "champion"):
            try:
                shadow = self._model_predict(challenger, image)
                self.shadow_agreements.append(
                    shadow.label == prediction.label)
            except Exception:  # noqa: BLE001 - shadow must never fail a request
                pass

        prediction.served_by = arm
        prediction.latency_ms = latency_ms
        prediction.ood_distance = ood_dist
        prediction.ood_flag = ood_flag
        prediction.z_outlier_rate = z_rate
        prediction.drift_flag = drift_flag
        self.served[arm] += 1
        return prediction

    def predict_batch(self, images: list[list[list[float]]],
                      request_ids: list[str] | None = None
                      ) -> list[Prediction]:
        return [self.predict(img, request_id=(request_ids[i]
                                              if request_ids else None))
                for i, img in enumerate(images)]

    def stats(self) -> dict:
        champion = self.registry.get_champion()
        ref_edges = champion.ref_bin_edges if champion else []
        shadow_n = len(self.shadow_agreements)
        return {
            "served": dict(self.served),
            "fallbacks": self.fallbacks,
            "shadow_agreement_rate": (
                round(sum(self.shadow_agreements) / shadow_n, 4)
                if shadow_n else None),
            "shadow_n": shadow_n,
            "drift": self.drift.summary(ref_edges),
            "alerts": self.drift.alerts,
        }
