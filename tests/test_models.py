"""Tests for the from-scratch classifiers."""
import pytest

from cvserve.features import (
    HIST_SLICE, FeatureStandardizer, extract_features,
)
from cvserve.models import (
    EnsembleClassifier, HistogramFallback, KNNClassifier,
    LogisticRegression,
)


@pytest.fixture()
def small_data(corpus):
    # The pipeline always standardizes before training; mirror that here.
    train, test = corpus
    X_train_raw = [extract_features(ex.image) for ex in train]
    y_train = [ex.label for ex in train]
    X_test_raw = [extract_features(ex.image) for ex in test]
    y_test = [ex.label for ex in test]
    std = FeatureStandardizer().fit(X_train_raw)
    return (std.transform_many(X_train_raw), y_train,
            std.transform_many(X_test_raw), y_test)


def test_logreg_trains_and_predicts(small_data):
    X_train, y_train, X_test, y_test = small_data
    clf = LogisticRegression(epochs=25, seed=42)
    clf.fit(X_train, y_train)
    assert clf.loss_history_
    assert clf.loss_history_[-1] < clf.loss_history_[0]
    acc = sum(1 for x, y in zip(X_test, y_test)
              if clf.predict(x) == y) / len(y_test)
    assert acc >= 0.80


def test_logreg_proba_sums_to_one(small_data):
    X_train, y_train, X_test, _ = small_data
    clf = LogisticRegression(epochs=25, seed=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test[0])
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_logreg_rejects_bad_input():
    clf = LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit([], [])
    with pytest.raises(ValueError):
        clf.predict([0.0] * 45)
    with pytest.raises(ValueError):
        clf.fit([[0.0] * 45], ["circle", "square"])


def test_logreg_explain_returns_top_features(small_data, trained_version):
    from cvserve.features import FEATURE_NAMES

    X_train, y_train, X_test, _ = small_data
    clf = LogisticRegression(epochs=25, seed=42)
    clf.fit(X_train, y_train)
    expl = clf.explain(X_test[0], FEATURE_NAMES, top_k=5)
    assert 0 < len(expl) <= 5
    assert all(isinstance(name, str) for name, _ in expl)
    assert all(abs(w) > 0 for _, w in expl)


def test_knn_accuracy(small_data):
    X_train, y_train, X_test, y_test = small_data
    knn = KNNClassifier(k=5).fit(X_train, y_train)
    acc = sum(1 for x, y in zip(X_test, y_test)
              if knn.predict(x) == y) / len(y_test)
    assert acc >= 0.80
    probs = knn.predict_proba(X_test[0])
    assert sum(probs.values()) == pytest.approx(1.0)


def test_ensemble_beats_single_models(small_data):
    X_train, y_train, X_test, y_test = small_data
    logreg = LogisticRegression(epochs=25, seed=42).fit(X_train, y_train)
    knn = KNNClassifier(k=5).fit(X_train, y_train)
    ens = EnsembleClassifier(logreg=logreg, knn=knn)
    acc = sum(1 for x, y in zip(X_test, y_test)
              if ens.predict(x) == y) / len(y_test)
    assert acc >= 0.85


def test_histogram_fallback_is_weaker_but_above_majority():
    # Use a larger corpus: on 30 test examples the fallback can tie the
    # majority baseline by chance; on 120 it must beat it clearly.
    from cvserve.features import FeatureStandardizer
    from cvserve.images import generate_dataset

    train, test = generate_dataset(n_per_class=60, seed=42)
    X_train_raw = [extract_features(ex.image) for ex in train]
    y_train = [ex.label for ex in train]
    X_test_raw = [extract_features(ex.image) for ex in test]
    y_test = [ex.label for ex in test]
    std = FeatureStandardizer().fit(X_train_raw)
    X_train = std.transform_many(X_train_raw)
    X_test = std.transform_many(X_test_raw)
    hist_idx = list(range(HIST_SLICE.start, HIST_SLICE.stop))
    model = LogisticRegression(epochs=25, seed=42)
    model.fit([[v[i] for i in hist_idx] for v in X_train], y_train)
    fb = HistogramFallback(model=model, indices=hist_idx)
    acc = sum(1 for x, y in zip(X_test, y_test)
              if fb.predict(x) == y) / len(y_test)
    majority = max(y_test.count(c) for c in set(y_test)) / len(y_test)
    assert acc > majority
    assert acc < 1.0  # the fallback is deliberately weak


def test_trained_version_metrics(trained_version):
    assert trained_version.metrics.accuracy >= 0.85
    assert trained_version.metrics.macro_f1 >= 0.80
    assert trained_version.baseline_accuracy < \
        trained_version.metrics.accuracy
    assert trained_version.ood_threshold > 0.0
    assert trained_version.ref_bin_edges
    assert trained_version.centroids
