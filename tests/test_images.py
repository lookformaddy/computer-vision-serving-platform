"""Tests for the synthetic image generator and augmentation."""
import random

import pytest

from cvserve.images import (
    CLASSES, SIZE, augment, flip_horizontal, flip_vertical, generate_dataset,
    generate_image, resize_nearest, rotate90, translate,
)


def test_generate_image_is_deterministic():
    a = generate_image("circle", 123)
    b = generate_image("circle", 123)
    assert a == b


def test_generate_image_varies_with_seed():
    assert generate_image("circle", 1) != generate_image("circle", 2)


def test_generate_image_shape_and_range():
    for shape in CLASSES:
        img = generate_image(shape, 42)
        assert len(img) == SIZE
        assert all(len(row) == SIZE for row in img)
        assert all(0.0 <= v <= 1.0 for row in img for v in row)


def test_generate_image_unknown_shape_raises():
    with pytest.raises(ValueError):
        generate_image("hexagon", 1)


def test_classes_are_visually_distinct():
    # Mean images of different classes differ substantially.
    means = {}
    for shape in CLASSES:
        img = generate_image(shape, 42)
        means[shape] = sum(sum(row) for row in img) / (SIZE * SIZE)
    # Checkerboard is ~half foreground; shapes are mostly background.
    assert means["checker"] > means["circle"]
    assert means["checker"] > 0.40
    assert means["triangle"] < 0.30
    assert means["circle"] != means["square"]


def test_dataset_split_and_labels(corpus):
    train, test = corpus
    assert train and test
    assert len(train) + len(test) == 30 * len(CLASSES)
    assert {ex.label for ex in train + test} <= set(CLASSES)
    # Test labels are clean: every test image matches its generator class.
    for ex in test:
        assert ex.image == generate_image(ex.label, ex.seed)


def test_flip_twice_is_identity(sample_image):
    assert flip_horizontal(flip_horizontal(sample_image)) == sample_image
    assert flip_vertical(flip_vertical(sample_image)) == sample_image


def test_rotate90_four_times_is_identity(sample_image):
    assert rotate90(sample_image, 4) == sample_image


def test_rotate90_preserves_total_intensity(sample_image):
    before = sum(sum(row) for row in sample_image)
    after = sum(sum(row) for row in rotate90(sample_image, 1))
    assert before == pytest.approx(after)


def test_translate_shifts_pixels():
    img = [[0.0] * SIZE for _ in range(SIZE)]
    img[5][5] = 1.0
    out = translate(img, 2, 3, fill=0.0)
    assert out[8][7] == 1.0
    assert out[5][5] == 0.0


def test_resize_nearest():
    img = generate_image("square", 7)
    small = resize_nearest(img, 16)
    assert len(small) == 16 and all(len(r) == 16 for r in small)
    big = resize_nearest(img, 48)
    assert len(big) == 48 and all(len(r) == 48 for r in big)


def test_augment_preserves_size_and_range(sample_image):
    rng = random.Random(0)
    out = augment(sample_image, rng)
    assert len(out) == SIZE and all(len(r) == SIZE for r in out)
    assert all(0.0 <= v <= 1.0 for row in out for v in row)


def test_augment_is_deterministic_for_rng():
    rng1 = random.Random(5)
    rng2 = random.Random(5)
    img = generate_image("star", 9)
    assert augment(img, rng1) == augment(img, rng2)
