"""Generate synthetic color-identification dataset (sanity-check task).

Each image contains 1–3 shapes. We pick one as the *target* and ask the model
to identify its color. The probe should easily succeed if representations
capture visual information at all.

This validates the full pipeline (extraction → probing) before we move to
the harder spatial-relation task.
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
    SHAPE_COLORS,
    ShapeInstance,
    ShapeType,
    ColorSample,
)
from .renderer import render_image


CANVAS_W = 448
CANVAS_H = 448
MARGIN = 60
SIZE_RANGE = (30, 55)

PROMPT_TEMPLATES_COLOR = [
    "The color of the {shape} in the image is",
    "What color is the {shape}? The answer is",
    "Looking at the image, the {shape} is colored",
]


def _random_positions(n: int, rng: random.Random) -> list[tuple[float, float, float]]:
    """Generate n non-overlapping (cx, cy, size) tuples."""
    positions = []
    for _ in range(n * 20):  # retry budget
        s = rng.uniform(*SIZE_RANGE)
        cx = rng.uniform(MARGIN + s, CANVAS_W - MARGIN - s)
        cy = rng.uniform(MARGIN + s, CANVAS_H - MARGIN - s)
        # Check overlap
        ok = True
        for ox, oy, os in positions:
            if abs(cx - ox) < (s + os + 15) and abs(cy - oy) < (s + os + 15):
                ok = False
                break
        if ok:
            positions.append((cx, cy, s))
        if len(positions) == n:
            break
    return positions


def generate_color_dataset(
    n_samples: int = 1000,
    output_dir: str = "data/raw/color",
    seed: int = 123,
    min_shapes: int = 1,
    max_shapes: int = 3,
    prompt_template_index: Optional[int] = None,  # None = random
) -> list[dict]:
    """Generate color-identification images and metadata."""
    rng = random.Random(seed)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    color_names = list(SHAPE_COLORS.keys())
    shape_types = list(ShapeType)

    # Ensure roughly balanced color distribution for the *target*
    targets_per_color = n_samples // len(color_names)
    color_list = []
    for c in color_names:
        color_list.extend([c] * targets_per_color)
    while len(color_list) < n_samples:
        color_list.append(rng.choice(color_names))
    rng.shuffle(color_list)

    samples: list[dict] = []

    for i, target_color in enumerate(tqdm(color_list, desc="Generating color dataset")):
        sample_id = f"color_{i:05d}"
        n_shapes = rng.randint(min_shapes, max_shapes)
        positions = _random_positions(n_shapes, rng)

        if len(positions) < 1:
            # fallback: single centered shape
            positions = [(CANVAS_W / 2, CANVAS_H / 2, 40)]
            n_shapes = 1

        # Assign the target color to a random shape index
        target_idx = rng.randint(0, len(positions) - 1)

        shapes = []
        for j, (cx, cy, s) in enumerate(positions):
            stype = rng.choice(shape_types)
            if j == target_idx:
                cname = target_color
            else:
                # Pick a different color from the target to avoid ambiguity
                other_colors = [c for c in color_names if c != target_color]
                cname = rng.choice(other_colors)
            shapes.append(ShapeInstance(stype, cname, cx, cy, s))

        target_shape = shapes[target_idx]

        # Render
        img = render_image(shapes, canvas_size=(CANVAS_W, CANVAS_H))
        img_filename = f"{sample_id}.png"
        img.save(img_dir / img_filename)

        # Prompt — always reference shape type so the model knows which shape
        # When multiple shapes: use color-free descriptor like "the circle"
        # When one shape: "the shape" or "the <shape_type>"
        if n_shapes == 1:
            desc = f"the {target_shape.shape_type.value}"
        else:
            # Use shape type; if duplicates exist, also mention position
            desc = f"the {target_shape.shape_type.value}"

        if prompt_template_index is not None:
            template = PROMPT_TEMPLATES_COLOR[prompt_template_index]
        else:
            template = rng.choice(PROMPT_TEMPLATES_COLOR)
        prompt = template.format(shape=desc)

        sample = ColorSample(
            image_id=sample_id,
            image_filename=img_filename,
            target_shape=target_shape,
            color_label=target_color,
            prompt=prompt,
        )
        samples.append(sample.to_dict())

    # Save metadata
    meta_path = Path(output_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} color samples → {output_dir}")
    return samples