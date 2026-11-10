"""Tests for the versioned registry and promotion gates."""
import copy

import pytest

from cvserve.pipeline import train_version
from cvserve.registry import ModelRegistry


@pytest.fixture()
def two_versions():
    v1, _ = train_version("v1", seed=42, n_per_class=30, epochs=25)
    v2, _ = train_version("v2", seed=7, n_per_class=30, epochs=25)
    return v1, v2


def test_register_champion_and_challenger(two_versions):
    v1, v2 = two_versions
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    reg.register(v2, role="challenger")
    assert reg.champion == "v1"
    assert reg.challenger == "v2"
    assert reg.get_champion().version == "v1"
    assert len(reg) == 2
    assert len(reg.list_versions()) == 2


def test_register_duplicate_raises(two_versions):
    v1, _ = two_versions
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    with pytest.raises(ValueError):
        reg.register(v1, role="challenger")


def test_register_bad_role_raises(two_versions):
    v1, _ = two_versions
    reg = ModelRegistry()
    with pytest.raises(ValueError):
        reg.register(v1, role="shadow")


def test_promote_good_challenger_passes(two_versions):
    v1, v2 = two_versions
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    reg.register(v2, role="challenger")
    result = reg.promote("v2", v1.baseline_accuracy)
    assert result.passed
    assert reg.champion == "v2"
    assert reg.challenger == "v1"
    assert any(h["action"] == "promote" for h in reg.history)


def test_promote_fails_when_accuracy_below_floor(two_versions):
    v1, v2 = two_versions
    bad = copy.deepcopy(v2)
    bad.metrics.accuracy = 0.50
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    reg.register(bad, role="challenger")
    result = reg.promote(bad.version, v1.baseline_accuracy)
    assert not result.passed
    assert not result.checks["accuracy_floor"]
    assert reg.champion == "v1"
    assert any(h["action"] == "promote_attempt" and not h["passed"]
               for h in reg.history)


def test_promote_fails_when_it_does_not_beat_baseline(two_versions):
    v1, v2 = two_versions
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    reg.register(v2, role="challenger")
    # Baseline set impossibly high: the challenger cannot beat it.
    result = reg.promote("v2", 0.99)
    assert not result.passed
    assert not result.checks["beats_baseline"]
    assert reg.champion == "v1"


def test_promote_fails_on_latency_ceiling(two_versions):
    v1, v2 = two_versions
    slow = copy.deepcopy(v2)
    slow.p95_latency_ms = 999.0
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    reg.register(slow, role="challenger")
    result = reg.promote(slow.version, v1.baseline_accuracy)
    assert not result.passed
    assert not result.checks["latency_ceiling"]


def test_promote_unknown_or_already_champion_raises(two_versions):
    v1, v2 = two_versions
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    with pytest.raises(ValueError):
        reg.promote("nope", 0.5)
    with pytest.raises(ValueError):
        reg.promote("v1", 0.5)


def test_rollback_restores_prior_champion(two_versions):
    v1, v2 = two_versions
    reg = ModelRegistry()
    reg.register(v1, role="champion")
    reg.register(v2, role="challenger")
    reg.promote("v2", v1.baseline_accuracy)
    assert reg.champion == "v2"
    reg.rollback("v1")
    assert reg.champion == "v1"
    with pytest.raises(ValueError):
        reg.rollback("nope")
