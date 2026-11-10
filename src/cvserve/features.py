"""Hand-built image feature extraction (pure stdlib, no numpy).

A 45-dimensional feature vector per image, in a fixed order. Every
block is invariant to rotation (and, except where noted, translation
and scale) — the synthetic corpus varies rotation, scale, and
translation per image, so rotation-sensitive features (raw zonal means,
raw orientation bins) would be within-class noise:

- ring statistics: 4 concentric rings x (mean, std, p10, p90) (16)
- intensity histogram: 8 bins over [0, 1] (8)
- gradient summaries: orientation entropy / top-bin shares, gradient
  magnitude mean/std/p10/p90, edge fraction, gradient energy,
  centre gradient mean (10)
- moment/shape invariants: Hu moments 1-4 (log-transformed, clamped),
  fill ratio, perimeter ratio, log connected-component count (7)
- global intensity stats: mean, std, foreground fraction, median (4)

``FeatureStandardizer`` z-scores features with training-set statistics;
every model in this project consumes standardized features.
"""
from __future__ import annotations

import math

__all__ = [
    "FEATURE_NAMES", "N_FEATURES", "HIST_SLICE", "extract_features",
    "hu_moments", "radial_profile", "FeatureStandardizer",
]

N_RING = 16
N_HIST = 8
N_GRAD = 10
N_MOMENTS = 7
N_GLOBAL = 4
N_FEATURES = N_RING + N_HIST + N_GRAD + N_MOMENTS + N_GLOBAL  # 45

# Indices of the histogram-only slice (used by the fallback model).
HIST_SLICE = slice(N_RING, N_RING + N_HIST)

FEATURE_NAMES: list[str] = (
    [f"ring{r}_{stat}" for r in range(4)
     for stat in ("mean", "std", "p10", "p90")]
    + [f"hist_{i}" for i in range(N_HIST)]
    + ["orient_entropy", "orient_max", "orient_second",
       "grad_mean", "grad_std", "grad_p90", "grad_p10",
       "edge_frac", "grad_energy", "center_grad_mean"]
    + ["hu_1", "hu_2", "hu_3", "hu_4",
       "fill_ratio", "perimeter_ratio", "log_components"]
    + ["global_mean", "global_std", "frac_fg", "median_intensity"]
)
assert len(FEATURE_NAMES) == N_FEATURES


def _ring_values(image: list[list[float]]) -> list[list[float]]:
    """Pixel values grouped into 4 concentric rings around the centre."""
    size = len(image)
    cx = cy = (size - 1) / 2.0
    max_r = math.hypot(cx, cy) or 1.0
    rings: list[list[float]] = [[] for _ in range(4)]
    for y in range(size):
        for x in range(size):
            r = math.hypot(x - cx, y - cy) / max_r
            rings[min(int(r * 4), 3)].append(image[y][x])
    return rings


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[idx]


def _ring_stats(image: list[list[float]]) -> list[float]:
    feats: list[float] = []
    for ring in _ring_values(image):
        if not ring:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        ordered = sorted(ring)
        mean = sum(ring) / len(ring)
        var = sum((v - mean) ** 2 for v in ring) / len(ring)
        feats.append(mean)
        feats.append(math.sqrt(var))
        feats.append(_percentile(ordered, 0.10))
        feats.append(_percentile(ordered, 0.90))
    return feats


def radial_profile(image: list[list[float]]) -> list[float]:
    """Mean intensity in 4 concentric rings around the image centre."""
    return [sum(ring) / len(ring) if ring else 0.0
            for ring in _ring_values(image)]


def _histogram(image: list[list[float]]) -> list[float]:
    hist = [0.0] * N_HIST
    total = 0
    for row in image:
        for v in row:
            hist[min(int(v * N_HIST), N_HIST - 1)] += 1
            total += 1
    return [h / total for h in hist] if total else hist


def _gradient_features(image: list[list[float]]) -> list[float]:
    size = len(image)
    cx = cy = (size - 1) / 2.0
    max_r = math.hypot(cx, cy) or 1.0
    orient = [0.0] * 8
    mags: list[float] = []
    center_mags: list[float] = []
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            gx = ((image[y - 1][x + 1] + 2 * image[y][x + 1]
                   + image[y + 1][x + 1])
                  - (image[y - 1][x - 1] + 2 * image[y][x - 1]
                     + image[y + 1][x - 1]))
            gy = ((image[y + 1][x - 1] + 2 * image[y + 1][x]
                   + image[y + 1][x + 1])
                  - (image[y - 1][x - 1] + 2 * image[y - 1][x]
                     + image[y - 1][x + 1]))
            mag = math.hypot(gx, gy)
            mags.append(mag)
            if math.hypot(x - cx, y - cy) / max_r < 0.25:
                center_mags.append(mag)
            if mag > 1e-9:
                ang = math.atan2(gy, gx) % math.pi
                orient[min(int(ang / math.pi * 8), 7)] += mag
    total = sum(orient) or 1.0
    shares = [o / total for o in orient]
    entropy = -sum(p * math.log(p) for p in shares if p > 0.0)
    ranked = sorted(shares, reverse=True)
    if mags:
        ordered = sorted(mags)
        mean_mag = sum(mags) / len(mags)
        var = sum((m - mean_mag) ** 2 for m in mags) / len(mags)
        grad_mean = mean_mag
        grad_std = math.sqrt(var)
        grad_p90 = _percentile(ordered, 0.90)
        grad_p10 = _percentile(ordered, 0.10)
        edge_frac = sum(1 for m in mags if m > 1.0) / len(mags)
        grad_energy = sum(m * m for m in mags) / len(mags)
    else:
        grad_mean = grad_std = grad_p90 = grad_p10 = 0.0
        edge_frac = grad_energy = 0.0
    center_grad = (sum(center_mags) / len(center_mags)
                   if center_mags else 0.0)
    return [entropy, ranked[0], ranked[1] if len(ranked) > 1 else 0.0,
            grad_mean, grad_std, grad_p90, grad_p10, edge_frac,
            grad_energy, center_grad]


def hu_moments(image: list[list[float]]) -> list[float]:
    """Hu invariant moments 1-4, log-transformed and clamped.

    Pixel values are treated as mass. The transform is
    ``-sign(h) * log10(|h|)``, the standard scale-compressed form,
    clamped to [-10, 10]; a zero moment maps to 0.0. Moments 5-7 are
    deliberately omitted: they are numerically unstable on noisy,
    discrete images (they swing by orders of magnitude between seeds
    of the same class) and would drown the stable signal.
    """
    size = len(image)
    m00 = m10 = m01 = 0.0
    for y in range(size):
        for x in range(size):
            v = image[y][x]
            m00 += v
            m10 += x * v
            m01 += y * v
    if m00 == 0.0:
        return [0.0] * 4
    xbar = m10 / m00
    ybar = m01 / m00
    mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 = 0.0
    for y in range(size):
        for x in range(size):
            v = image[y][x]
            dx = x - xbar
            dy = y - ybar
            mu20 += dx * dx * v
            mu11 += dx * dy * v
            mu02 += dy * dy * v
            mu30 += dx ** 3 * v
            mu21 += dx * dx * dy * v
            mu12 += dx * dy * dy * v
            mu03 += dy ** 3 * v
    eta20 = mu20 / m00 ** 2
    eta11 = mu11 / m00 ** 2
    eta02 = mu02 / m00 ** 2
    eta30 = mu30 / m00 ** 2.5
    eta21 = mu21 / m00 ** 2.5
    eta12 = mu12 / m00 ** 2.5
    eta03 = mu03 / m00 ** 2.5
    hu = [
        eta20 + eta02,
        (eta20 - eta02) ** 2 + 4 * eta11 ** 2,
        (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2,
        (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2,
    ]
    out: list[float] = []
    for h in hu:
        if abs(h) <= 1e-12:
            out.append(0.0)
        else:
            out.append(max(-10.0, min(10.0,
                                      -math.copysign(math.log10(abs(h)), h))))
    return out


def _shape_invariants(image: list[list[float]]) -> list[float]:
    """Fill ratio, perimeter ratio, and log connected-component count
    of the foreground (pixels > 0.5), 4-connected."""
    size = len(image)
    fg = [[v > 0.5 for v in row] for row in image]
    n_fg = sum(sum(row) for row in fg)
    total = size * size
    fill_ratio = n_fg / total if total else 0.0
    if n_fg == 0:
        return [0.0, 0.0, 0.0]
    perimeter = 0
    for y in range(size):
        for x in range(size):
            if not fg[y][x]:
                continue
            if (x == 0 or not fg[y][x - 1] or x == size - 1
                    or not fg[y][x + 1] or y == 0 or not fg[y - 1][x]
                    or y == size - 1 or not fg[y + 1][x]):
                perimeter += 1
    perimeter_ratio = perimeter / n_fg
    # 4-connected components of the foreground (flood fill).
    seen = [[False] * size for _ in range(size)]
    components = 0
    for y0 in range(size):
        for x0 in range(size):
            if not fg[y0][x0] or seen[y0][x0]:
                continue
            components += 1
            stack = [(x0, y0)]
            seen[y0][x0] = True
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x + 1, y), (x - 1, y),
                               (x, y + 1), (x, y - 1)):
                    if (0 <= nx < size and 0 <= ny < size
                            and fg[ny][nx] and not seen[ny][nx]):
                        seen[ny][nx] = True
                        stack.append((nx, ny))
    return [fill_ratio, perimeter_ratio, math.log1p(components)]


def _moment_features(image: list[list[float]]) -> list[float]:
    return hu_moments(image) + _shape_invariants(image)


def _global_stats(image: list[list[float]]) -> list[float]:
    vals = sorted(v for row in image for v in row)
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    var = sum((v - mean) ** 2 for v in vals) / n if n else 0.0
    frac_fg = sum(1 for v in vals if v > 0.5) / n if n else 0.0
    median = vals[n // 2] if n else 0.0
    return [mean, math.sqrt(var), frac_fg, median]


def extract_features(image: list[list[float]]) -> list[float]:
    """Full 45-dimensional feature vector for one image."""
    if not image or not image[0]:
        raise ValueError("cannot extract features from an empty image")
    feats: list[float] = []
    feats.extend(_ring_stats(image))
    feats.extend(_histogram(image))
    feats.extend(_gradient_features(image))
    feats.extend(_moment_features(image))
    feats.extend(_global_stats(image))
    assert len(feats) == N_FEATURES, f"expected {N_FEATURES}, got {len(feats)}"
    return feats


class FeatureStandardizer:
    """Per-feature z-score standardization from training statistics."""

    def __init__(self) -> None:
        self.means_: list[float] = []
        self.stds_: list[float] = []

    def fit(self, X: list[list[float]]) -> "FeatureStandardizer":
        if not X:
            raise ValueError("cannot fit standardizer on empty data")
        n = len(X)
        d = len(X[0])
        self.means_ = [sum(row[j] for row in X) / n for j in range(d)]
        self.stds_ = []
        for j in range(d):
            mean = self.means_[j]
            var = sum((row[j] - mean) ** 2 for row in X) / n
            self.stds_.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return self

    def transform(self, vec: list[float]) -> list[float]:
        if not self.means_:
            raise ValueError("standardizer is not fitted")
        return [(v - m) / s for v, m, s in
                zip(vec, self.means_, self.stds_)]

    def transform_many(self, X: list[list[float]]) -> list[list[float]]:
        return [self.transform(vec) for vec in X]
