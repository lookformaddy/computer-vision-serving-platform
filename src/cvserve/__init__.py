"""computer-vision-serving-platform: production image-classification
serving, built from scratch (pure stdlib, no numpy/torch/sklearn).

A deterministic synthetic shape/texture corpus, hand-built feature
extraction (zonal, histogram, gradient-orientation, Hu moments, radial
profile), a from-scratch multinomial logistic-regression + k-NN
ensemble, and the operational machinery a real serving platform needs:
versioned registry with promotion gates, canary/shadow deployment,
deterministic hash A/B routing, latency budgets with a lightweight
histogram-only fallback, feature-drift (PSI/KS) and OOD monitoring.
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
