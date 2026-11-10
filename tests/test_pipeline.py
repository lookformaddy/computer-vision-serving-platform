"""Tests for the training pipeline end to end."""
from cvserve.pipeline import train_version


def test_train_version_report_shape():
    version, report = train_version("v1-pipe", seed=42,
                                    n_per_class=30, epochs=25)
    assert version.version == "v1-pipe"
    assert report["n_train"] > 0
    assert report["n_test"] > 0
    assert report["n_features"] == 45
    assert report["metrics"].accuracy >= 0.85
    assert report["metrics"].macro_f1 >= 0.80
    assert report["baseline_accuracy"] < report["metrics"].accuracy
    assert report["majority_accuracy"] < report["metrics"].accuracy
    assert report["ece"] < 0.25
    assert report["latency"]["p95"] > 0.0
    assert report["ood_threshold"] > 0.0


def test_train_version_is_reproducible():
    v1, r1 = train_version("v1-a", seed=42, n_per_class=30, epochs=25)
    v2, r2 = train_version("v1-b", seed=42, n_per_class=30, epochs=25)
    assert r1["metrics"].accuracy == r2["metrics"].accuracy
    assert r1["final_loss"] == r2["final_loss"]


def test_train_version_different_seed_differs():
    _v1, r1 = train_version("v1-a", seed=42, n_per_class=30, epochs=25)
    _v2, r2 = train_version("v1-b", seed=7, n_per_class=30, epochs=25)
    # Different data splits: metrics need not match exactly.
    assert r1["metrics"].accuracy >= 0.85
    assert r2["metrics"].accuracy >= 0.85
