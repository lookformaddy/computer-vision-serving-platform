# Architecture — computer-vision-serving-platform

## Problem

An image classifier that only works in a notebook is not a product: it
needs versioned models, safe rollout, a fallback when it is slow or
wrong, drift detection when the incoming image distribution shifts
(new camera, new lighting, new backgrounds), and an explanation for
every decision an operator acts on. This project builds that whole
serving surface around a from-scratch vision classifier — no sklearn,
no numpy, no torch — so every layer is inspectable.

## Diagram (description)

```
                ┌─────────────────────┐
   seed ──────▶ │  images.py          │  deterministic synthetic 32x32
                │  (generation)       │  shapes/textures + augmentation
                └─────────┬───────────┘  (flips, 90° rotations,
                          ▼               brightness, translation)
                ┌─────────────────────┐
                │  features.py        │  ring stats 16 · histogram 8 ·
                │  extract_features   │  gradient summaries 10 ·
                │  (45 dims)          │  Hu 1-4 + shape invariants 7 ·
                │                     │  global intensity stats 4
                └─────────┬───────────┘
                          ▼
                ┌─────────────────────┐
                │  FeatureStandardizer│  train-set z-scores
                └─────────┬───────────┘
                          ▼
                ┌─────────────────────┐
                │  models.py          │  multinomial logistic regression
                │  EnsembleClassifier │  (full-batch GD, softmax + L2)
                │  logreg + k-NN      │  + k-NN, probability-averaged
                └─────────┬───────────┘
                          ▼
   ┌──────────────────────────────────────────────────────┐
   │  registry.py — ModelRegistry                         │
   │  versions (v1, v2, ...) · champion / challenger      │
   │  promotion gates: accuracy floor · beats-baseline    │
   │  margin · per-class F1 floor · p95 latency ceiling · │
   │  full decision history · rollback                    │
   └──────────────────────┬───────────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────────┐
   │  serving.py — ServingLayer                           │
   │  drift monitor (OOD distance-to-centroid, z-outliers,│
   │  rolling-window PSI vs training deciles) ──▶         │
   │  hash A/B routing (sha256(request_id) % 100          │
   │  < canary_pct)                                       │
   │        ├─ champion / challenger arm                  │
   │        ├─ shadow scoring of the challenger +         │
   │        │  agreement rate (never returned)            │
   │        └─ latency budget ──▶ histogram-only          │
   │           fallback (fast, always available)          │
   │  every response: label, confidence, top-feature      │
   │  explanation, arm, latency, OOD/drift flags          │
   └──────────────────────┬───────────────────────────────┘
                          ▼
        ┌─────────────────┴─────────────────┐
        │ service.py (FastAPI)              │ cli.py
        │ /health /predict /models          │ train / eval /
        │ /stats /promote                   │ predict / batch / serve
        └───────────────────────────────────┘
   evaluation.py — accuracy, per-class P/R/F1, confusion matrix,
   ECE calibration, latency percentiles, histogram-fallback and
   majority baselines (the gates' floors and the fallback)
```

## Key design decisions

1. **Stdlib-only core.** The generator, features, classifiers, and
   routing are pure Python. That is a deliberate trade: training is
   slower than a numpy build, but the code is auditable line by line,
   has zero binary-dependency risk in the serving image, and the
   dense 45-feature representation is exactly what a low-latency
   serving path wants.
2. **Hand-built features, not raw pixels.** Ring statistics,
   histogram, gradient summaries, Hu moments, and shape invariants are
   rotation-invariant by design (the corpus varies rotation, scale,
   and translation per image, so rotation-sensitive features would be
   within-class noise), interpretable, make the linear model
   explainable (`feature × weight`), and keep k-NN distances
   meaningful. A CNN would learn features at the cost of opacity and
   a heavyweight dependency; the fixed feature set is the known
   ceiling of this design.
3. **Deterministic hash routing.** `sha256(request_id) % 100` makes the
   canary arm stable across retries for the same request — the property
   that makes A/B results interpretable. No request id means champion,
   so ad-hoc traffic never leaks into an experiment.
4. **Promotion gates, not auto-promotion.** A challenger must clear an
   accuracy floor, beat the histogram baseline by a margin, pass a
   per-class F1 floor, and stay under a p95 latency ceiling before it
   can become champion. Every attempt is logged with its gate results,
   and `rollback()` restores any prior version.
5. **Fallback is a feature, not an exception path.** When the model
   path exceeds its latency budget or raises, the histogram-only model
   answers instead — degraded but always available — and the fallback
   is counted and reported in `/stats`.
6. **Shadow before canary.** A challenger first scores champion
   traffic in shadow mode (agreement rate recorded, result discarded);
   only then does canary traffic reach it. That ordering catches
   behavioral regressions offline metrics miss.
7. **Drift as a first-class signal.** Every request is scored for OOD
   distance (to the nearest class centroid in standardized feature
   space, thresholded at the 99th percentile of training distances)
   and z-score outliers; the rolling window's per-feature PSI is
   checked against the training decile reference, with alerts when PSI
   exceeds 0.2 or the OOD rate exceeds 10%.

## Trade-offs

- Full-batch gradient descent converges reliably on this corpus size
  but would need mini-batching (or a different optimizer) past ~10k
  images; the training loop is the known scaling bottleneck.
- The histogram fallback is intentionally weak — it is a floor and a
  fallback, not a competitor. Its accuracy sets the bar the learned
  model must clear by a margin.
- Explanations are linear feature contributions (`feature × weight`).
  They are faithful for the logistic-regression member by
  construction, but they do not capture the k-NN member's vote or
  feature interactions.
- OOD detection is distance-to-centroid in a 45-d standardized space.
  It catches gross distribution shift (noise, corruption) reliably;
  subtle per-class drift is the PSI window's job, not the OOD score's.

## Data

All training and test images are synthetic, generated deterministically
from seeds (see `images.py`). A configurable label-noise rate flips a
small fraction of training labels to mimic annotation noise; test
labels are always clean. No real images, no employer data, no external
datasets.

## Running it

```bash
pip install -r requirements.txt
make test        # pytest suite (API tests skip without FastAPI)
make demo        # train -> evaluate -> promote -> serve examples
python -m cvserve.cli predict --shape star --seed 404
python -m cvserve.cli serve --port 8000   # then POST /predict
docker build -t computer-vision-serving-platform . && docker run -p 8000:8000 computer-vision-serving-platform
```

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
tests/           pytest suite (API tests auto-skip without FastAPI)
docs/            architecture.md, github-actions-ci.yml
```
