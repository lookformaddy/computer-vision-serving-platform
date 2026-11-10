"""Tests for the serving layer: routing, fallback, shadow, drift."""
import random

import pytest

from cvserve.images import generate_image
from cvserve.pipeline import train_version
from cvserve.registry import ModelRegistry
from cvserve.serving import ServingConfig, ServingLayer, hash_arm


def test_hash_arm_is_deterministic_and_bounded():
    assert hash_arm("req-1") == hash_arm("req-1")
    assert 0 <= hash_arm("req-1") < 100
    assert hash_arm("req-1") != hash_arm("req-2")


def test_predict_returns_label_and_explanation(serving, sample_image):
    pred = serving.predict(sample_image, request_id="t-1")
    assert pred.label in ("circle", "square", "triangle", "star",
                          "checker")
    assert 0.0 < pred.confidence <= 1.0
    assert pred.explanation
    assert pred.served_by in ("champion", "challenger", "fallback")
    assert pred.latency_ms >= 0.0
    assert pred.model_version == "v1-test"


def test_predict_without_request_id_goes_to_champion(serving, sample_image):
    pred = serving.predict(sample_image)
    assert pred.served_by == "champion"


def test_canary_routing_is_deterministic(serving, sample_image):
    # With a challenger registered and canary_pct=100, every hashed
    # request goes to the challenger, stably.
    v2, _ = train_version("v2-test", seed=7, n_per_class=30, epochs=25)
    serving.registry.register(v2, role="challenger")
    serving.config.canary_pct = 100
    first = serving.predict(sample_image, request_id="stable-1")
    second = serving.predict(sample_image, request_id="stable-1")
    assert first.served_by == "challenger"
    assert first.served_by == second.served_by
    assert first.label == second.label


def test_shadow_agreement_recorded(serving, sample_image):
    v2, _ = train_version("v2-test", seed=7, n_per_class=30, epochs=25)
    serving.registry.register(v2, role="challenger")
    serving.config.canary_pct = 0  # all traffic to champion -> shadow runs
    serving.predict(sample_image, request_id="shadow-1")
    serving.predict(generate_image("square", 55), request_id="shadow-2")
    assert len(serving.shadow_agreements) == 2
    stats = serving.stats()
    assert stats["shadow_n"] == 2
    assert 0.0 <= stats["shadow_agreement_rate"] <= 1.0


def test_latency_budget_falls_back(serving, sample_image):
    serving.config.latency_budget_ms = 0.0  # impossible budget
    pred = serving.predict(sample_image, request_id="slow-1")
    assert pred.served_by == "fallback"
    assert pred.fallback_reason is not None
    assert "budget" in pred.fallback_reason
    assert serving.fallbacks == 1
    assert serving.served["fallback"] == 1


def test_ood_noise_image_is_flagged(serving):
    rng = random.Random(1234)
    noise = [[rng.random() for _ in range(32)] for _ in range(32)]
    pred = serving.predict(noise, request_id="ood-1")
    assert pred.ood_flag
    assert pred.drift_flag
    assert pred.ood_distance > 0.0


def test_in_distribution_image_is_not_ood(serving, sample_image):
    pred = serving.predict(sample_image, request_id="in-1")
    assert not pred.ood_flag


def test_predict_batch(serving):
    images = [generate_image("circle", i) for i in range(5)]
    preds = serving.predict_batch(images, request_ids=[f"b-{i}"
                                                       for i in range(5)])
    assert len(preds) == 5
    assert sum(1 for p in preds if p.label == "circle") >= 4


def test_stats_counts_served_arms(serving, sample_image):
    serving.predict(sample_image, request_id="s-1")
    serving.predict(sample_image, request_id="s-2")
    stats = serving.stats()
    assert stats["served"]["champion"] == 2
    assert stats["drift"]["n_requests"] == 2


def test_no_champion_raises():
    layer = ServingLayer(ModelRegistry())
    with pytest.raises(ValueError):
        layer.predict(generate_image("circle", 1))
