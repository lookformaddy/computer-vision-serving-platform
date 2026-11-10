"""Deterministic synthetic image generation and augmentation.

Every image is generated in-process from a seed — no external datasets,
no downloaded files, no employer data. Five classes of 32x32 grayscale
images (values in [0, 1]):

    circle, square, triangle, star  — filled geometric shapes with
        random rotation, scale, translation, and foreground/background
        intensity; edges are hard (no anti-aliasing) so gradient
        features stay discriminative.
    checker                         — a 4x4 checkerboard texture with
        per-cell intensity jitter (a texture class, not a shape).

Generation is fully deterministic: ``generate_image(shape, seed)``
always returns the same pixels. A small label-noise rate in
:func:`generate_dataset` flips a fraction of *training* labels to mimic
annotation noise; test labels are always clean.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "CLASSES", "SIZE", "Image", "Example", "generate_image",
    "generate_dataset", "flip_horizontal", "flip_vertical", "rotate90",
    "translate", "adjust_brightness", "resize_nearest", "augment",
]

CLASSES: list[str] = ["circle", "square", "triangle", "star", "checker"]
SIZE: int = 32

# An image is a list of rows; each row is a list of floats in [0, 1].
Image = list[list[float]]


@dataclass(frozen=True)
class Example:
    image: Image
    label: str
    seed: int


def _point_in_triangle(px: float, py: float,
                       ax: float, ay: float, bx: float, by: float,
                       cx: float, cy: float) -> bool:
    def sign(x1: float, y1: float, x2: float, y2: float,
             x3: float, y3: float) -> float:
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _point_in_polygon(px: float, py: float,
                      verts: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _shape_vertices(shape: str, radius: float) -> list[tuple[float, float]]:
    """Polygon vertices for the shape, centred on the origin."""
    if shape == "triangle":
        return [(radius * math.cos(-math.pi / 2 + k * 2 * math.pi / 3),
                 radius * math.sin(-math.pi / 2 + k * 2 * math.pi / 3))
                for k in range(3)]
    if shape == "star":
        verts: list[tuple[float, float]] = []
        for k in range(10):
            r = radius if k % 2 == 0 else radius * 0.45
            ang = -math.pi / 2 + k * math.pi / 5
            verts.append((r * math.cos(ang), r * math.sin(ang)))
        return verts
    raise ValueError(f"no polygon vertices for shape {shape!r}")


def generate_image(shape: str, seed: int, size: int = SIZE,
                   noise_sigma: float = 0.05) -> Image:
    """Generate one synthetic image deterministically from ``seed``."""
    if shape not in CLASSES:
        raise ValueError(f"unknown shape {shape!r}; expected one of "
                         f"{CLASSES}")
    rng = random.Random(seed)
    fg = rng.uniform(0.75, 0.95)
    bg = rng.uniform(0.05, 0.20)
    half = (size - 1) / 2.0

    if shape == "checker":
        # 4x4 checkerboard texture with per-cell intensity jitter.
        cells = 4
        cell = size // cells
        phase = rng.randrange(2)
        cell_val: dict[tuple[int, int], float] = {}
        for r in range(cells):
            for c in range(cells):
                base = fg if (r + c + phase) % 2 == 0 else bg
                cell_val[(r, c)] = base * rng.uniform(0.85, 1.0)
        img: Image = []
        for y in range(size):
            row: list[float] = []
            for x in range(size):
                v = cell_val[(y // cell, x // cell)]
                v += rng.gauss(0.0, noise_sigma)
                row.append(min(1.0, max(0.0, v)))
            img.append(row)
        return img

    # Geometric shapes: random centre offset, scale, rotation.
    dx = rng.uniform(-3.0, 3.0)
    dy = rng.uniform(-3.0, 3.0)
    radius = rng.uniform(0.55, 0.90) * (size * 0.42)
    theta = rng.uniform(0.0, 2 * math.pi)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    verts = _shape_vertices(shape, radius) if shape in (
        "triangle", "star") else None

    img = []
    for y in range(size):
        row = []
        for x in range(size):
            # Pixel centre in shape-local coordinates.
            lx = x - half - dx
            ly = y - half - dy
            rx = lx * cos_t - ly * sin_t
            ry = lx * sin_t + ly * cos_t
            if shape == "circle":
                inside = rx * rx + ry * ry <= radius * radius
            elif shape == "square":
                inside = abs(rx) <= radius and abs(ry) <= radius
            elif verts is not None:
                if shape == "triangle":
                    (ax, ay), (bx, by), (cx, cy) = verts
                    inside = _point_in_triangle(rx, ry, ax, ay, bx, by,
                                                cx, cy)
                else:
                    inside = _point_in_polygon(rx, ry, verts)
            else:  # pragma: no cover - guarded by the CLASSES check
                inside = False
            v = fg if inside else bg
            v += rng.gauss(0.0, noise_sigma)
            row.append(min(1.0, max(0.0, v)))
        img.append(row)
    return img


def generate_dataset(n_per_class: int = 100, seed: int = 42,
                     test_fraction: float = 0.2,
                     label_noise: float = 0.02
                     ) -> tuple[list[Example], list[Example]]:
    """Seeded train/test split of synthetic images.

    Train labels carry a small configurable noise rate (annotation-noise
    simulation); test labels are always clean.
    """
    rng = random.Random(seed)
    train: list[Example] = []
    test: list[Example] = []
    for label in CLASSES:
        for _ in range(n_per_class):
            img_seed = rng.randrange(1 << 30)
            image = generate_image(label, img_seed)
            noisy_label = label
            if rng.random() < label_noise:
                noisy_label = rng.choice(
                    [c for c in CLASSES if c != label])
            example = Example(image=image, label=noisy_label,
                              seed=img_seed)
            if rng.random() < test_fraction:
                # Test examples keep the clean label.
                test.append(Example(image=image, label=label,
                                    seed=img_seed))
            else:
                train.append(example)
    return train, test


# -- augmentation / preprocessing --------------------------------------

def flip_horizontal(image: Image) -> Image:
    return [row[::-1] for row in image]


def flip_vertical(image: Image) -> Image:
    return image[::-1]


def rotate90(image: Image, k: int = 1) -> Image:
    """Rotate clockwise by 90 degrees, ``k`` times."""
    out = image
    for _ in range(k % 4):
        out = [list(reversed(col)) for col in zip(*out)]
    return out


def translate(image: Image, dx: int, dy: int, fill: float = 0.1) -> Image:
    size = len(image)
    out: Image = []
    for y in range(size):
        src_y = y - dy
        row: list[float] = []
        for x in range(size):
            src_x = x - dx
            if 0 <= src_y < size and 0 <= src_x < size:
                row.append(image[src_y][src_x])
            else:
                row.append(fill)
        out.append(row)
    return out


def adjust_brightness(image: Image, factor: float) -> Image:
    return [[min(1.0, max(0.0, v * factor)) for v in row]
            for row in image]


def resize_nearest(image: Image, new_size: int) -> Image:
    """Nearest-neighbour resize to ``new_size`` x ``new_size``."""
    size = len(image)
    out: Image = []
    for y in range(new_size):
        src_y = min(size - 1, int(y * size / new_size))
        row: list[float] = []
        for x in range(new_size):
            src_x = min(size - 1, int(x * size / new_size))
            row.append(image[src_y][src_x])
        out.append(row)
    return out


def augment(image: Image, rng: random.Random) -> Image:
    """Random augmentation: flips, 90-degree rotations, brightness,
    small integer translation. Deterministic for a given ``rng``."""
    out = image
    if rng.random() < 0.5:
        out = flip_horizontal(out)
    if rng.random() < 0.5:
        out = flip_vertical(out)
    out = rotate90(out, rng.randrange(4))
    out = adjust_brightness(out, rng.uniform(0.85, 1.15))
    out = translate(out, rng.randint(-2, 2), rng.randint(-2, 2))
    return out
