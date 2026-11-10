"""Tests for offline evaluation metrics."""
import pytest

from cvserve.evaluation import (
    ClassificationMetrics, evaluate, expected_calibration_error,
    latency_percentiles, majority_baseline_accuracy,
)


def _perfect_probs(labels):
    return [{lab: 1.0 if lab == t else 0.0 for lab in labels}
            for t in ["circle", "square", "circle", "square"]]


def test_evaluate_perfect_predictions():
    labels = ["circle", "square"]
    metrics = evaluate(_perfect_probs(labels),
                       ["circle", "square", "circle", "square"])
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.confusion[("circle", "circle")] == 2
    assert metrics.confusion[("square", "square")] == 2


def test_evaluate_counts_errors():
    probs = [
        {"circle": 0.9, "square": 0.1},
        {"circle": 0.2, "square": 0.8},
        {"circle": 0.8, "square": 0.2},  # wrong: true square
        {"circle": 0.1, "square": 0.9},
    ]
    metrics = evaluate(probs, ["circle", "square", "square", "square"])
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.confusion[("square", "circle")] == 1
    assert metrics.per_class["circle"]["recall"] == pytest.approx(1.0)
    assert metrics.per_class["square"]["recall"] == pytest.approx(2 / 3)


def test_evaluate_rejects_mismatched_and_empty():
    with pytest.raises(ValueError):
        evaluate([{"circle": 1.0}], [])
    with pytest.raises(ValueError):
        evaluate([], [])


def test_confusion_table_renders():
    labels = ["circle", "square"]
    metrics = evaluate(_perfect_probs(labels),
                       ["circle", "square", "circle", "square"])
    table = metrics.confusion_table()
    assert "circle" in table and "square" in table
    assert "2" in table


def test_ece_perfect_is_zero():
    labels = ["circle", "square"]
    ece = expected_calibration_error(
        _perfect_probs(labels), ["circle", "square", "circle", "square"])
    assert ece == pytest.approx(0.0)


def test_ece_overconfident_is_positive():
    probs = [{"circle": 0.99, "square": 0.01}] * 4
    ece = expected_calibration_error(probs, ["square"] * 4)
    assert ece > 0.5


def test_latency_percentiles():
    lat = latency_percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert lat["p50"] == pytest.approx(3.0)
    assert lat["p95"] == pytest.approx(100.0)
    assert lat["max"] == pytest.approx(100.0)
    assert lat["mean"] == pytest.approx(22.0)


def test_latency_percentiles_empty():
    assert latency_percentiles([])["p95"] == 0.0


def test_majority_baseline():
    assert majority_baseline_accuracy(
        ["circle", "circle", "square"]) == pytest.approx(2 / 3)
    assert majority_baseline_accuracy([]) == 0.0
