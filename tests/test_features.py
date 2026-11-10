"""Tests for hand-built feature extraction."""
import pytest

from cvserve.features import (
    FEATURE_NAMES, HIST_SLICE, N_FEATURES, FeatureStandardizer,
    extract_features, hu_moments, radial_profile,
)
from cvserve.images import generate_image


def test_feature_vector_length_and_names(features):
    assert len(features) == N_FEATURES == 45
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_features_are_deterministic(sample_image):
    assert extract_features(sample_image) == extract_features(sample_image)


def test_features_differ_across_classes():
    feats = {shape: extract_features(generate_image(shape, 42))
             for shape in ("circle", "square", "triangle", "star",
                           "checker")}
    for a in feats:
        for b in feats:
            if a < b:
                assert feats[a] != feats[b]


def test_histogram_slice_sums_to_one(features):
    hist = features[HIST_SLICE]
    assert len(hist) == 8
    assert sum(hist) == pytest.approx(1.0)


def test_ring_means_in_range(features):
    ring_means = features[0:16:4]
    assert len(ring_means) == 4
    assert all(0.0 <= v <= 1.0 for v in ring_means)


def test_orientation_summaries_are_bounded(features):
    # orient_entropy, orient_max, orient_second (indices 24-26).
    entropy, top, second = features[24], features[25], features[26]
    assert 0.0 <= entropy <= 2.1  # ln(8) max
    assert 0.0 < top <= 1.0
    assert 0.0 <= second <= top


def test_hu_moments_are_finite_and_clamped(sample_image):
    hu = hu_moments(sample_image)
    assert len(hu) == 4
    assert all(v == v and -10.0 <= v <= 10.0 for v in hu)


def test_radial_profile_length_and_range(sample_image):
    profile = radial_profile(sample_image)
    assert len(profile) == 4
    assert all(0.0 <= v <= 1.0 for v in profile)


def test_checker_has_many_components():
    feats = extract_features(generate_image("checker", 42))
    # log_components is index 40 (16 ring + 8 hist + 10 grad + 4 hu + 2).
    import math

    assert feats[40] == pytest.approx(math.log1p(8), abs=0.6)


def test_shape_has_one_component():
    feats = extract_features(generate_image("circle", 42))
    # One foreground component (log1p(1)); allow a stray noise pixel.
    assert feats[40] == pytest.approx(0.6931, abs=0.45)


def test_empty_image_raises():
    with pytest.raises(ValueError):
        extract_features([])


def test_standardizer_zero_mean_unit_variance(corpus):
    from cvserve.features import extract_features as ef

    train, _test = corpus
    X = [ef(ex.image) for ex in train]
    std = FeatureStandardizer().fit(X)
    Xs = std.transform_many(X)
    n = len(Xs)
    for j in range(N_FEATURES):
        mean = sum(row[j] for row in Xs) / n
        assert mean == pytest.approx(0.0, abs=1e-9)


def test_standardizer_rejects_unfitted():
    std = FeatureStandardizer()
    with pytest.raises(ValueError):
        std.transform([0.0] * N_FEATURES)
