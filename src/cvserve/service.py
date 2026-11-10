"""FastAPI service for the computer-vision serving platform.

Endpoints:
    GET  /health    — liveness + champion version + drift summary
    POST /predict   — classify one image (by generated shape+seed, or by
                      raw 32x32 pixels); champion/canary/fallback arms
    GET  /models    — registry versions, champion, challenger
    GET  /stats     — serving stats: arms, fallbacks, shadow agreement, drift
    POST /promote   — promote the challenger if the registry gates pass

The module-level ``app`` is built at import with a champion trained on
the seeded corpus, so ``TestClient(app)`` works out of the box in tests.
For production wiring use :func:`build_app` with your own ServingLayer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException

from cvserve.features import extract_features
from cvserve.images import CLASSES, SIZE, generate_image
from cvserve.pipeline import train_version
from cvserve.registry import ModelRegistry
from cvserve.serving import ServingConfig, ServingLayer

__all__ = ["app", "build_app"]


class PredictRequest(BaseModel):
    shape: str | None = Field(
        default=None,
        description="Synthetic shape class to generate and classify")
    seed: int | None = Field(
        default=None, description="Seed for the generated image")
    pixels: list[float] | None = Field(
        default=None,
        description=f"Flat {SIZE}x{SIZE} grayscale pixels in [0, 1]")
    request_id: str | None = Field(
        default=None,
        description="Stable id for deterministic canary/A-B routing")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[tuple[str, float]]
    served_by: str
    model_version: str | None
    latency_ms: float
    fallback_reason: str | None = None
    ood_distance: float
    ood_flag: bool
    drift_flag: bool


class PromoteRequest(BaseModel):
    baseline_accuracy: float = Field(..., ge=0.0, le=1.0)


def _request_image(req: PredictRequest) -> list[list[float]]:
    if req.pixels is not None:
        if len(req.pixels) != SIZE * SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"pixels must contain {SIZE * SIZE} values")
        if any(v < 0.0 or v > 1.0 for v in req.pixels):
            raise HTTPException(status_code=422,
                                detail="pixels must be in [0, 1]")
        return [req.pixels[i * SIZE:(i + 1) * SIZE] for i in range(SIZE)]
    if req.shape is not None:
        if req.shape not in CLASSES:
            raise HTTPException(status_code=422,
                                detail=f"unknown shape {req.shape!r}")
        return generate_image(req.shape, req.seed if req.seed is not None
                              else 7)
    raise HTTPException(status_code=422,
                        detail="provide either 'shape' or 'pixels'")


def build_app(serving: ServingLayer) -> FastAPI:
    app = FastAPI(title="computer-vision-serving-platform",
                  description="Production image-classification serving "
                              "(from-scratch ensemble + registry + "
                              "canary/shadow + drift monitoring)")

    @app.get("/health")
    def health() -> dict:
        champion = serving.registry.get_champion()
        return {
            "status": "ok",
            "champion": champion.version if champion else None,
            "challenger": serving.registry.challenger,
            "drift": serving.stats()["drift"],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        image = _request_image(req)
        result = serving.predict(image, request_id=req.request_id)
        champion = serving.registry.get_champion()
        vec = (champion.standardizer.transform(extract_features(image))
               if champion else [])
        probs = (champion.classifier.predict_proba(vec)
                 if champion else {})
        return PredictResponse(
            label=result.label,
            confidence=result.confidence,
            probabilities=probs,
            explanation=result.explanation,
            served_by=result.served_by,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
            fallback_reason=result.fallback_reason,
            ood_distance=result.ood_distance,
            ood_flag=result.ood_flag,
            drift_flag=result.drift_flag,
        )

    @app.get("/models")
    def models() -> dict:
        return {
            "champion": serving.registry.champion,
            "challenger": serving.registry.challenger,
            "versions": serving.registry.list_versions(),
        }

    @app.get("/stats")
    def stats() -> dict:
        return serving.stats()

    @app.post("/promote")
    def promote(req: PromoteRequest) -> dict:
        challenger = serving.registry.get_challenger()
        if challenger is None:
            raise HTTPException(status_code=409,
                                detail="no challenger registered")
        result = serving.registry.promote(challenger.version,
                                         req.baseline_accuracy)
        return {
            "passed": result.passed,
            "checks": result.checks,
            "details": result.details,
            "champion": serving.registry.champion,
        }

    return app


def _default_serving() -> ServingLayer:
    version, _report = train_version("v1-seed42", seed=42,
                                     n_per_class=60, epochs=60)
    registry = ModelRegistry()
    registry.register(version, role="champion")
    return ServingLayer(registry, ServingConfig(canary_pct=10))


app = build_app(_default_serving())
