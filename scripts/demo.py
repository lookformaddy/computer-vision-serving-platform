"""End-to-end demo: train, evaluate, promote, serve a few examples.

Trains a champion (v1) and a challenger (v2, different seed), runs the
offline eval, checks the promotion gates, then serves example images
through the full serving layer — canary routing, shadow scoring,
latency budgets, drift flags, and feature explanations included. An
out-of-distribution noise image and a heavily corrupted image show the
OOD detector and the histogram-only fallback.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvserve.images import CLASSES, generate_image  # noqa: E402
from cvserve.pipeline import train_version  # noqa: E402
from cvserve.registry import ModelRegistry  # noqa: E402
from cvserve.serving import ServingConfig, ServingLayer  # noqa: E402

EXAMPLES = [
    ("circle", 101),
    ("square", 202),
    ("triangle", 303),
    ("star", 404),
    ("checker", 505),
]


def _noise_image(seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.random() for _ in range(32)] for _ in range(32)]


def _corrupted_image() -> list[list[float]]:
    # A circle image drowned in noise: same shape, destroyed signal.
    rng = random.Random(777)
    base = generate_image("circle", 606)
    return [[min(1.0, max(0.0, v + rng.gauss(0.0, 0.35)))
             for v in row] for row in base]


def main() -> int:
    print("== computer-vision-serving-platform demo ==\n")

    champion, report = train_version("v1-seed42", seed=42)
    metrics = report["metrics"]
    print(f"[train] champion v1-seed42: {report['n_train']} train / "
          f"{report['n_test']} test, features={report['n_features']}, "
          f"final loss={report['final_loss']:.4f}")
    print(metrics.summary())
    print(f"[eval] histogram-fallback baseline accuracy: "
          f"{report['baseline_accuracy']:.4f}")
    print(f"[eval] majority baseline accuracy: "
          f"{report['majority_accuracy']:.4f}")
    print(f"[eval] ECE={report['ece']:.4f} "
          f"p50={report['latency']['p50']:.2f}ms "
          f"p95={report['latency']['p95']:.2f}ms "
          f"p99={report['latency']['p99']:.2f}ms")
    print(f"[eval] OOD threshold: {report['ood_threshold']:.3f}")
    print(metrics.confusion_table())

    challenger, report2 = train_version("v2-seed7", seed=7)
    print(f"\n[train] challenger v2-seed7: "
          f"accuracy={report2['metrics'].accuracy:.4f} "
          f"macro_f1={report2['metrics'].macro_f1:.4f}")

    registry = ModelRegistry()
    registry.register(champion, role="champion")
    registry.register(challenger, role="challenger")
    result = registry.promote("v2-seed7", report["baseline_accuracy"])
    print(f"\n[promote] challenger v2-seed7:\n{result}")

    # A fresh challenger (v3) takes the challenger slot so the serve
    # step exercises canary routing + shadow scoring for real.
    challenger_v3, report3 = train_version("v3-seed99", seed=99)
    registry.register(challenger_v3, role="challenger")
    print(f"[train] new challenger v3-seed99: "
          f"accuracy={report3['metrics'].accuracy:.4f} "
          f"macro_f1={report3['metrics'].macro_f1:.4f}")

    serving = ServingLayer(registry, ServingConfig(canary_pct=25))
    print("\n[serve] example images (canary 25%):")
    for i, (shape, seed) in enumerate(EXAMPLES):
        image = generate_image(shape, seed)
        pred = serving.predict(image, request_id=f"demo-{i}")
        feats = ", ".join(f"{n} ({w:+.2f})"
                          for n, w in pred.explanation[:3])
        mark = " ✓" if pred.label == shape else f" ✗ (expected {shape})"
        print(f"  {shape:<9} -> {pred.label:<9} conf={pred.confidence:.2f} "
              f"via={pred.served_by:<10} {pred.latency_ms:5.1f}ms "
              f"ood={pred.ood_distance:.2f} drift={pred.drift_flag} "
              f"[{feats}]{mark}")

    # Out-of-distribution: pure noise, and a noise-corrupted circle.
    for name, image in (("noise (OOD)", _noise_image(1234)),
                        ("corrupted circle", _corrupted_image())):
        pred = serving.predict(image, request_id=f"demo-ood-{name}")
        print(f"  {name:<18} -> {pred.label:<9} conf={pred.confidence:.2f} "
              f"via={pred.served_by:<10} {pred.latency_ms:5.1f}ms "
              f"ood={pred.ood_distance:.2f} drift={pred.drift_flag}")

    print("\n[stats]", serving.stats())
    print("\nDemo complete: train -> evaluate -> promote -> serve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
