"""Offline evaluation: accuracy, per-class precision/recall/F1,
confusion matrix, calibration (ECE), latency percentiles, and baselines
(majority class, histogram-only fallback) the promotion gates must beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ClassificationMetrics", "evaluate", "expected_calibration_error",
    "latency_percentiles", "majority_baseline_accuracy",
]


@dataclass
class ClassificationMetrics:
    accuracy: float
    per_class: dict[str, dict[str, float]]  # label -> p/r/f1/support
    macro_f1: float
    confusion: dict[tuple[str, str], int]   # (true, pred) -> count
    labels: list[str] = field(default_factory=list)
    n: int = 0

    def summary(self) -> str:
        lines = [f"accuracy={self.accuracy:.4f} "
                 f"macro_f1={self.macro_f1:.4f} n={self.n}"]
        for lab in self.labels:
            m = self.per_class[lab]
            lines.append(
                f"  {lab:<10} P={m['precision']:.3f} "
                f"R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"(support {m['support']})")
        return "\n".join(lines)

    def confusion_table(self) -> str:
        header = "true \\ pred | " + " ".join(f"{lab[:6]:>6}"
                                              for lab in self.labels)
        rows = [header]
        for true in self.labels:
            cells = " ".join(
                f"{self.confusion.get((true, pred), 0):6d}"
                for pred in self.labels)
            rows.append(f"{true[:10]:<10} | {cells}")
        return "\n".join(rows)


def evaluate(probs_list: list[dict[str, float]],
             y_true: list[str]) -> ClassificationMetrics:
    """Metrics from per-example probability dicts and true labels."""
    if len(probs_list) != len(y_true):
        raise ValueError("probs and labels must have the same length")
    if not y_true:
        raise ValueError("cannot evaluate on empty data")
    labels = sorted(set(y_true))
    y_pred = [max(p, key=lambda lab: p[lab]) for p in probs_list]
    confusion: dict[tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        confusion[(t, p)] = confusion.get((t, p), 0) + 1
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    per_class: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = confusion.get((lab, lab), 0)
        fp = sum(v for (t, p), v in confusion.items()
                 if p == lab and t != lab)
        fn = sum(v for (t, p), v in confusion.items()
                 if t == lab and p != lab)
        support = sum(v for (t, _p), v in confusion.items() if t == lab)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[lab] = {"precision": precision, "recall": recall,
                          "f1": f1, "support": float(support)}
    macro_f1 = (sum(m["f1"] for m in per_class.values()) / len(labels)
                if labels else 0.0)
    return ClassificationMetrics(
        accuracy=correct / len(y_true), per_class=per_class,
        macro_f1=macro_f1, confusion=confusion, labels=labels,
        n=len(y_true))


def expected_calibration_error(probs_list: list[dict[str, float]],
                               y_true: list[str],
                               n_bins: int = 10) -> float:
    """ECE: mean |mean confidence - empirical accuracy| per bin."""
    if not y_true:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for probs, true in zip(probs_list, y_true):
        pred = max(probs, key=lambda lab: probs[lab])
        conf = probs[pred]
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, pred == true))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / len(y_true)) * abs(mean_conf - acc)
    return ece


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """p50 / p95 / p99 / mean / max of per-request latencies (ms)."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0,
                "max": 0.0}
    ordered = sorted(latencies_ms)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n)))
        return ordered[idx]

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "mean": sum(ordered) / n, "max": ordered[-1]}


def majority_baseline_accuracy(y_true: list[str]) -> float:
    """Accuracy of always predicting the majority class."""
    if not y_true:
        return 0.0
    counts: dict[str, int] = {}
    for lab in y_true:
        counts[lab] = counts.get(lab, 0) + 1
    return max(counts.values()) / len(y_true)
