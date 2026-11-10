"""Command-line interface: train / eval / predict / batch / serve.

Examples:
    python -m cvserve.cli train --version v1 --seed 42
    python -m cvserve.cli predict --shape circle --seed 7
    python -m cvserve.cli batch --n 20 --seed 99
    python -m cvserve.cli serve --port 8000
"""
from __future__ import annotations

import argparse
import random
import sys

from cvserve.images import CLASSES, generate_image
from cvserve.pipeline import train_version
from cvserve.registry import ModelRegistry
from cvserve.serving import ServingConfig, ServingLayer

__all__ = ["main"]


def _build_serving(seed: int = 42, canary_pct: int = 10) -> ServingLayer:
    version, _report = train_version("v1-seed42", seed=seed)
    registry = ModelRegistry()
    registry.register(version, role="champion")
    return ServingLayer(registry, ServingConfig(canary_pct=canary_pct))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cvserve",
        description="Computer-vision image-classification serving "
                    "(from-scratch ensemble + registry + canary/shadow)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train and evaluate a version")
    p_train.add_argument("--version", default="v1")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--n-per-class", type=int, default=100)
    p_train.add_argument("--epochs", type=int, default=120)

    p_eval = sub.add_parser("eval", help="train then report held-out metrics")
    p_eval.add_argument("--version", default="v1")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--n-per-class", type=int, default=100)
    p_eval.add_argument("--epochs", type=int, default=120)

    p_pred = sub.add_parser("predict", help="classify one synthetic image")
    p_pred.add_argument("--shape", choices=CLASSES, default="circle")
    p_pred.add_argument("--seed", type=int, default=7)
    p_pred.add_argument("--request-id", default=None)
    p_pred.add_argument("--canary-pct", type=int, default=10)

    p_batch = sub.add_parser("batch", help="classify N random images")
    p_batch.add_argument("--n", type=int, default=20)
    p_batch.add_argument("--seed", type=int, default=99)
    p_batch.add_argument("--canary-pct", type=int, default=10)

    p_serve = sub.add_parser("serve", help="start the FastAPI service")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command in ("train", "eval"):
        version, report = train_version(
            args.version, seed=args.seed, n_per_class=args.n_per_class,
            epochs=args.epochs)
        metrics = report["metrics"]
        print(f"[train] {args.version}: {report['n_train']} train / "
              f"{report['n_test']} test, features={report['n_features']}, "
              f"final loss={report['final_loss']:.4f}")
        print(metrics.summary())
        print(metrics.confusion_table())
        print(f"[eval] histogram-fallback baseline accuracy: "
              f"{report['baseline_accuracy']:.4f}")
        print(f"[eval] majority baseline accuracy: "
              f"{report['majority_accuracy']:.4f}")
        print(f"[eval] ECE={report['ece']:.4f} "
              f"p50={report['latency']['p50']:.2f}ms "
              f"p95={report['latency']['p95']:.2f}ms "
              f"p99={report['latency']['p99']:.2f}ms")
        print(f"[eval] OOD threshold (99th pct train distance): "
              f"{report['ood_threshold']:.3f}")
        return 0

    if args.command == "predict":
        serving = _build_serving(canary_pct=args.canary_pct)
        image = generate_image(args.shape, args.seed)
        pred = serving.predict(image, request_id=args.request_id)
        feats = ", ".join(f"{n} ({w:+.2f})"
                          for n, w in pred.explanation[:3])
        print(f"{pred.label} (conf={pred.confidence:.2f}) "
              f"via={pred.served_by} {pred.latency_ms:.1f}ms "
              f"ood={pred.ood_distance:.2f} drift={pred.drift_flag} "
              f"[{feats}] (true shape: {args.shape})")
        return 0

    if args.command == "batch":
        serving = _build_serving(canary_pct=args.canary_pct)
        rng = random.Random(args.seed)
        correct = 0
        for i in range(args.n):
            shape = rng.choice(CLASSES)
            image = generate_image(shape, rng.randrange(1 << 30))
            pred = serving.predict(image, request_id=f"batch-{i}")
            correct += int(pred.label == shape)
        print(f"batch accuracy: {correct}/{args.n} = {correct / args.n:.3f}")
        print("[stats]", serving.stats())
        return 0

    if args.command == "serve":
        import uvicorn  # noqa: PLC0415 - optional runtime dependency

        from cvserve.service import build_app  # noqa: PLC0415

        serving = _build_serving(seed=args.seed)
        app = build_app(serving)
        print(f"serving on http://{args.host}:{args.port} "
              f"(champion {serving.registry.champion})")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
