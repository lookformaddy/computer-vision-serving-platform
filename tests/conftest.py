"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvserve.features import extract_features  # noqa: E402
from cvserve.images import generate_dataset, generate_image  # noqa: E402
from cvserve.pipeline import train_version  # noqa: E402
from cvserve.registry import ModelRegistry  # noqa: E402
from cvserve.serving import ServingConfig, ServingLayer  # noqa: E402

SEED = 42


@pytest.fixture()
def corpus():
    return generate_dataset(n_per_class=30, seed=SEED)


@pytest.fixture()
def sample_image():
    return generate_image("circle", 101)


@pytest.fixture()
def features(sample_image):
    return extract_features(sample_image)


@pytest.fixture()
def trained_version():
    version, _report = train_version("v1-test", seed=SEED,
                                     n_per_class=30, epochs=25)
    return version


@pytest.fixture()
def registry(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    return reg


@pytest.fixture()
def serving(registry):
    return ServingLayer(registry, ServingConfig(canary_pct=25))
