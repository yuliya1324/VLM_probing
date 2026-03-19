"""Generate synthetic shape-identification dataset.

Each image contains exactly one shape. The model must identify the shape type.

Labels: circular, oval, square, rectangular, triangular

Robustness features:
  - Varied colors (random shades) so probes can't shortcut on color
  - Varied sizes so probes can't shortcut on size
  - Varied backgrounds
  - Varied positions (not always centered)
  - Oval vs circle and rectangle vs square are the tricky pairs —
    we ensure clear aspect ratios to avoid ambiguity
"""

import json
import random
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .schema import (
    COLOR_BASE,
    COLOR_SHADES,
    BACKGROUND_COLORS,
    SHAPE_TASK_TYPES,
    SHAPE_LABELS,
    ShapeInstance,
    ShapeType,
    ShapeSample,
)
from .renderer import render_image


CANVAS_W = 448
CANVAS_H = 448
MARGIN = 70
SIZE_RANGE = (30, 60)

PROMPT_TEMPLATES_SHAPE = [
    "The shape of the {color} object in the image is",
    "What shape is the {color} object? The answer is",
    "Identify the shape of the {color} object. It is",
    "Looking at the image, the {color} object is shaped like a",
]


def _pick_shade(rng: random.Random, color_name: str) -> tuple:
    """Pick a random shade for a color name."""
    shades = COLOR_SHADES.get(color_name)
    if shades:
        return rng.choice(shades)
    return COLOR_BASE[color_name]


def _pick_background(rng: random.Random, shape_color: str) -> tuple:
    """Pick a background that contrasts with the shape color."""
    candidates = list(BACKGROUND_COLORS)

    if shape_color == "white":
        candidates = [c for c in candidates if sum(c) / 3 < 210]
        if not candidates:
            candidates = [(160, 160, 160)]
    elif shape_color == "black":
        candidates = [c for c in candidates if sum(c) / 3 > 120]
        if not candidates:
            candidates = [(230, 230, 230)]

    return rng.choice(candidates)


def generate_shape_dataset(
    n_samples: int = 1000,
    output_dir: str = "data/raw/synthetic/shape",
    seed: int = 456,
) -> list:
    """Generate shape-identification images and metadata."""
    rng = random.Random(seed)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    color_names = list(COLOR_BASE.keys())
    shape_types = SHAPE_TASK_TYPES
    labels = [SHAPE_LABELS[st] for st in shape_types]

    # Balanced shape distribution
    per_shape = n_samples // len(shape_types)
    shape_list = []
    for st in shape_types:
        shape_list.extend([st] * per_shape)
    while len(shape_list) < n_samples:
        shape_list.append(rng.choice(shape_types))
    rng.shuffle(shape_list)

    samples = []

    for i, stype in enumerate(tqdm(shape_list, desc="Generating shape dataset")):
        sample_id = f"shape_{i:05d}"

        # Random color (avoid white/black on matching backgrounds)
        color_name = rng.choice(color_names)
        rgb = _pick_shade(rng, color_name)
        bg_color = _pick_background(rng, color_name)

        # Random size and position (not always centered)
        size = rng.uniform(*SIZE_RANGE)
        cx = rng.uniform(MARGIN + size, CANVAS_W - MARGIN - size)
        cy = rng.uniform(MARGIN + size, CANVAS_H - MARGIN - size)

        shape = ShapeInstance(stype, color_name, cx, cy, size, rgb_override=rgb)
        label = SHAPE_LABELS[stype]

        # Render single shape
        img = render_image([shape], canvas_size=(CANVAS_W, CANVAS_H), bg_color=bg_color)
        img_filename = f"{sample_id}.png"
        img.save(img_dir / img_filename)

        # Prompt — reference by color (since there's only one shape)
        template = rng.choice(PROMPT_TEMPLATES_SHAPE)
        prompt = template.format(color=color_name)

        sample = ShapeSample(
            image_id=sample_id,
            image_filename=img_filename,
            target_shape=shape,
            shape_label=label,
            prompt=prompt,
        )
        samples.append(sample.to_dict())

    # Save metadata
    meta_path = Path(output_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} shape samples → {output_dir}")
    return samples