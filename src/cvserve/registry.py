"""Versioned model registry with gated champion promotion.

A :class:`ModelVersion` bundles everything needed to serve one trained
model: the ensemble classifier, the histogram fallback, the feature
standardizer, the offline evaluation, the drift reference (per-feature
decile bin edges, per-class centroids, OOD threshold), and provenance.

Promotion is gated, never automatic. A challenger becomes champion only
if it clears every gate:

- accuracy floor (default 0.85 on held-out data)
- beats the histogram-fallback baseline by a margin (default +0.10)
- per-class F1 floor (default 0.70 for every class)
- p95 serving latency ceiling (default 150 ms)

Every promotion attempt — pass or fail — is recorded in the decision
history, and ``rollback`` restores any prior version as champion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from cvserve.evaluation import ClassificationMetrics
from cvserve.features import FeatureStandardizer
from cvserve.models import EnsembleClassifier, HistogramFallback

__all__ = ["ModelVersion", "PromotionResult", "ModelRegistry"]

ACCURACY_FLOOR = 0.85
BASELINE_MARGIN = 0.10
MIN_PER_CLASS_F1 = 0.70
P95_LATENCY_CEILING_MS = 150.0


@dataclass
class ModelVersion:
    version: str
    labels: list[str]
    standardizer: FeatureStandardizer
    classifier: EnsembleClassifier
    fallback: HistogramFallback
    metrics: ClassificationMetrics
    p95_latency_ms: float
    ece: float
    baseline_accuracy: float          # histogram-fallback accuracy
    majority_accuracy: float
    n_train: int
    n_test: int
    seed: int
    # Drift reference, computed from the training set:
    ref_bin_edges: list[list[float]] = field(default_factory=list)
    centroids: dict[str, list[float]] = field(default_factory=dict)
    ood_threshold: float = 0.0
    created_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def summary(self) -> dict:
        return {
            "version": self.version,
            "accuracy": round(self.metrics.accuracy, 4),
            "macro_f1": round(self.metrics.macro_f1, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "ece": round(self.ece, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "seed": self.seed,
            "created_at": self.created_at,
        }


@dataclass
class PromotionResult:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float | str]

    def __str__(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        lines = [f"promotion {verdict}"]
        for name, ok in self.checks.items():
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: "
                         f"{self.details.get(name, '')}")
        return "\n".join(lines)


@dataclass
class ModelRegistry:
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    champion: str | None = None
    challenger: str | None = None
    history: list[dict] = field(default_factory=list)

    def register(self, version: ModelVersion,
                 role: str = "challenger") -> None:
        if version.version in self.versions:
            raise ValueError(
                f"version {version.version!r} already registered")
        if role not in ("champion", "challenger"):
            raise ValueError("role must be 'champion' or 'challenger'")
        self.versions[version.version] = version
        if role == "champion":
            if self.champion is not None:
                self.challenger = self.champion
            self.champion = version.version
        else:
            self.challenger = version.version
        self.history.append({"action": "register",
                             "version": version.version, "role": role})

    def get_champion(self) -> ModelVersion | None:
        return self.versions.get(self.champion) if self.champion else None

    def get_challenger(self) -> ModelVersion | None:
        return (self.versions.get(self.challenger)
                if self.challenger else None)

    def list_versions(self) -> list[dict]:
        return [v.summary() for v in self.versions.values()]

    # -- promotion -------------------------------------------------------
    def promotion_checks(self, version: ModelVersion,
                         baseline_accuracy: float) -> PromotionResult:
        metrics = version.metrics
        min_f1 = min((m["f1"] for m in metrics.per_class.values()),
                     default=0.0)
        checks = {
            "accuracy_floor": metrics.accuracy >= ACCURACY_FLOOR,
            "beats_baseline": (
                metrics.accuracy >= baseline_accuracy + BASELINE_MARGIN),
            "per_class_f1_floor": min_f1 >= MIN_PER_CLASS_F1,
            "latency_ceiling": (
                version.p95_latency_ms <= P95_LATENCY_CEILING_MS),
        }
        details: dict[str, float | str] = {
            "accuracy_floor": round(metrics.accuracy, 4),
            "beats_baseline": round(metrics.accuracy - baseline_accuracy, 4),
            "per_class_f1_floor": round(min_f1, 4),
            "latency_ceiling": round(version.p95_latency_ms, 2),
        }
        return PromotionResult(passed=all(checks.values()), checks=checks,
                               details=details)

    def promote(self, version: str, baseline_accuracy: float
                ) -> PromotionResult:
        if version not in self.versions:
            raise ValueError(f"unknown version {version!r}")
        if version == self.champion:
            raise ValueError(f"version {version!r} is already champion")
        result = self.promotion_checks(self.versions[version],
                                       baseline_accuracy)
        self.history.append({
            "action": "promote_attempt", "version": version,
            "passed": result.passed, "checks": result.checks,
            "details": {k: str(v) for k, v in result.details.items()},
        })
        if result.passed:
            old = self.champion
            self.champion = version
            self.challenger = old if old != version else None
            self.history.append({"action": "promote",
                                 "version": version,
                                 "previous_champion": old or ""})
        return result

    def rollback(self, version: str) -> None:
        if version not in self.versions:
            raise ValueError(f"unknown version {version!r}")
        old = self.champion
        self.champion = version
        if self.challenger == version:
            self.challenger = old
        self.history.append({"action": "rollback", "version": version,
                             "previous_champion": old or ""})

    def __len__(self) -> int:
        return len(self.versions)
