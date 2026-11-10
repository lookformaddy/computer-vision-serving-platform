# computer-vision-serving-platform

Production computer-vision image-classification serving, built from
scratch — no sklearn, no numpy, no torch. A deterministic synthetic
shape/texture corpus, hand-built feature extraction (concentric-ring
statistics, intensity histogram, gradient summaries, Hu moments 1–4
plus shape invariants, global intensity stats), and a from-scratch multinomial logistic-regression + k-NN
ensemble, wrapped in the operational machinery a real serving platform
needs: a versioned model registry with promotion gates, canary + shadow
deployment, deterministic hash A/B routing, per-request latency budgets
with a lightweight histogram-only fallback, feature-drift (PSI) and
out-of-distribution monitoring, and a per-feature explanation for every
prediction.

## The enterprise problem

Image classifiers that only work in a notebook are not products. A
vision service in production needs versioned models with safe rollout,
a fallback when the model is slow or wrong, drift detection when the
incoming image distribution shifts (new camera, new lighting, new
backgrounds), and an explanation for every decision an operator acts
on. This project builds that whole serving surface around a classifier
whose every line is inspectable.

## What it demonstrates

- **From-scratch CV:** deterministic synthetic image generation
  (circle / square / triangle / star / checkerboard texture, with
  rotation, scale, translation, noise), hand-built features, and a
  logistic-regression + k-NN ensemble trained with full-batch gradient
  descent — pure stdlib, no ML framework.
- **Model lifecycle:** versioned registry, champion/challenger, and
  promotion gates (accuracy floor, must-beat-baseline margin,
  per-class F1 floor, p95 latency ceiling) with a full decision history
  and rollback.
- **Safe rollout:** deterministic `sha256(request_id)` canary routing,
  shadow scoring with agreement rate, and a latency budget that falls
  back to a histogram-only model instead of failing the request.
- **Observability:** feature-distribution drift (PSI over a rolling
  window), OOD detection via distance-to-centroid, z-score outlier
  rates, alerting, serving stats (arms, fallbacks, shadow agreement),
  and per-feature explanations (`standardized feature × weight`).
- **Honest evaluation:** accuracy, per-class precision/recall/F1, a
  confusion matrix, expected calibration error (ECE), and latency
  percentiles on held-out data, benchmarked against the histogram
  fallback and the majority-class baseline the promotion gates must
  beat.

## Architecture overview

```
synthetic image (32x32, seeded) → preprocessing (resize/normalize/augment)
        → features (ring stats 16 + histogram 8 + gradient 10 + moments 7 + global 4)
        → standardizer (train-set z-scores)
        → ensemble (logistic regression + k-NN, prob-averaged)
        → ModelRegistry (champion/challenger + promotion gates)
        → ServingLayer (drift → hash A/B routing → canary/shadow
                        → latency budget → histogram fallback)
        → FastAPI service (/health /predict /models /stats /promote)
          or CLI (train / eval / predict / batch / serve)
```

See [docs/architecture.md](docs/architecture.md) for the full diagram,
design decisions, and trade-offs.

## How to run it

```bash
# Install (core library needs only the stdlib; FastAPI/pytest are optional)
pip install -r requirements.txt

# Run the test suite (API tests auto-skip if FastAPI is absent)
make test

# End-to-end demo: train, evaluate, promote, serve examples
make demo

# Classify one synthetic image
PYTHONPATH=src python -m cvserve.cli predict --shape star --seed 404

# Train and evaluate a version
PYTHONPATH=src python -m cvserve.cli train --version v1 --seed 42

# Start the API
PYTHONPATH=src python -m cvserve.cli serve --port 8000
# curl -X POST localhost:8000/predict \
#   -H 'Content-Type: application/json' \
#   -d '{"shape": "triangle", "seed": 303, "request_id": "req-123"}'
# Or classify raw pixels (flat 32x32 list in [0, 1]):
# curl -X POST localhost:8000/predict \
#   -H 'Content-Type: application/json' \
#   -d '{"pixels": [...1024 values...]}'

# Docker
docker build -t computer-vision-serving-platform . && docker run -p 8000:8000 computer-vision-serving-platform
```

## Key design decisions and trade-offs

- **Stdlib-only core:** zero binary-dependency risk in the serving
  image and fully auditable code; the trade is slower training than a
  numpy build (fine at this corpus size; mini-batching is the known
  next step past ~10k images).
- **Hand-built features, not raw pixels:** 45 interpretable features
  keep the linear model explainable and the k-NN distances meaningful;
  the trade is that the feature set is fixed — a CNN would learn
  features, at the cost of opacity and a heavyweight dependency.
- **Deterministic hash routing:** the canary arm is stable across
  retries for the same request id, which is what makes A/B results
  interpretable; requests without an id always get the champion.
- **Gates, not auto-promotion:** a challenger must clear every gate
  before it becomes champion, and every attempt is recorded.
- **Fallback as a feature:** over-budget or erroring model calls fall
  back to the histogram-only model — degraded but always available —
  and every fallback is counted in `/stats`.
- **Shadow before canary:** the challenger first scores champion
  traffic in shadow mode (agreement recorded, result discarded), so
  behavioral regressions surface before any user sees the new model.
- **Drift as a first-class signal:** every request is scored for OOD
  distance and z-score outliers, and the rolling window's PSI is
  checked against the training reference, so distribution shift is
  visible before accuracy degrades.

## Data

All images are synthetic, generated deterministically from seeds
(`src/cvserve/images.py`), across five classes: `circle`, `square`,
`triangle`, `star`, `checker`. A configurable label-noise rate flips a
small fraction of training labels to mimic annotation noise; test
labels are always clean. No real images, no employer data, no external
datasets.

## Layout

```
src/cvserve/
  images.py      deterministic synthetic image generation + augmentation
  features.py    hand-built features (ring stats/histogram/gradient/
                 Hu 1-4 + shape invariants/global stats)
  models.py      logistic regression, k-NN, ensemble, histogram fallback
  evaluation.py  metrics, confusion matrix, ECE, latency percentiles
  registry.py    versioned registry + promotion gates
  serving.py     canary/shadow, routing, latency, fallback, drift (PSI/OOD)
  pipeline.py    train_version(): corpus -> model -> eval
  cli.py         train / eval / predict / batch / serve
  service.py     FastAPI app
scripts/demo.py  end-to-end demo
tests/           pytest suite
docs/            architecture.md, github-actions-ci.yml
```

The CI definition ships as `docs/github-actions-ci.yml` — move it to
`.github/workflows/ci.yml` to activate it (the push integration used
for this repo cannot write workflow files directly).
