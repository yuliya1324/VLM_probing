"""Generate synthetic color-identification dataset.

Each image contains 1–3 shapes on a varied background. We pick one shape
as the *target* and ask the model to identify its color.

Robustness features:
  - Color shades: each color name maps to multiple RGB values, randomly sampled
  - Background variation: random background colors to prevent shortcut learning
  - Target shape type is unique in the image (unambiguous prompts)
  - White/black shapes avoid white/light or dark backgrounds respectively
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
    "Identify the color of the {shape}. The color is",
]

# Backgrounds too close to white/black cause ambiguity
DARK_BG_THRESHOLD = 120   # avg RGB below this = "dark"
LIGHT_BG_THRESHOLD = 210  # avg RGB above this = "light"


def _pick_background(rng: random.Random, target_color: str) -> tuple:
    """Pick a background color that contrasts with the target."""
    candidates = list(BACKGROUND_COLORS)

    if target_color == "white":
        # Avoid light backgrounds
        candidates = [c for c in candidates if sum(c) / 3 < LIGHT_BG_THRESHOLD]
        if not candidates:
            candidates = [(160, 160, 160)]  # fallback medium gray
    elif target_color == "black":
        # Avoid dark backgrounds
        candidates = [c for c in candidates if sum(c) / 3 > DARK_BG_THRESHOLD]
        if not candidates:
            candidates = [(230, 230, 230)]  # fallback light gray

    return rng.choice(candidates)


def _pick_shade(rng: random.Random, color_name: str) -> tuple:
    """Pick a random shade for a color name."""
    shades = COLOR_SHADES.get(color_name)
    if shades:
        return rng.choice(shades)
    return COLOR_BASE[color_name]


def _random_positions(n: int, rng: random.Random) -> list:
    """Generate n non-overlapping (cx, cy, size) tuples."""
    positions = []
    for _ in range(n * 20):
        s = rng.uniform(*SIZE_RANGE)
        cx = rng.uniform(MARGIN + s, CANVAS_W - MARGIN - s)
        cy = rng.uniform(MARGIN + s, CANVAS_H - MARGIN - s)
        ok = True
        for ox, oy, os in positions:
            if abs(cx - ox) < (s + os + 20) and abs(cy - oy) < (s + os + 20):
                ok = False
                break
        if ok:
            positions.append((cx, cy, s))
        if len(positions) == n:
            break
    return positions


def generate_color_dataset(
    n_samples: int = 1000,
    output_dir: str = "data/raw/synthetic/color",
    seed: int = 123,
    min_shapes: int = 1,
    max_shapes: int = 3,
) -> list:
    """Generate color-identification images and metadata."""
    rng = random.Random(seed)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    color_names = list(COLOR_BASE.keys())
    shape_types = list(ShapeType)

    # Balanced color distribution
    targets_per_color = n_samples // len(color_names)
    color_list = []
    for c in color_names:
        color_list.extend([c] * targets_per_color)
    while len(color_list) < n_samples:
        color_list.append(rng.choice(color_names))
    rng.shuffle(color_list)

    samples = []

    for i, target_color in enumerate(tqdm(color_list, desc="Generating color dataset")):
        sample_id = f"color_{i:05d}"
        n_shapes = rng.randint(min_shapes, max_shapes)
        positions = _random_positions(n_shapes, rng)

        if len(positions) < 1:
            positions = [(CANVAS_W / 2, CANVAS_H / 2, 40)]

        # Pick background that contrasts with target color
        bg_color = _pick_background(rng, target_color)

        # Target gets a unique shape type
        target_idx = rng.randint(0, len(positions) - 1)
        target_stype = rng.choice(shape_types)
        other_stypes = [s for s in shape_types if s != target_stype]

        shapes = []
        for j, (cx, cy, s) in enumerate(positions):
            if j == target_idx:
                stype = target_stype
                cname = target_color
                rgb = _pick_shade(rng, target_color)
            else:
                stype = rng.choice(other_stypes)
                # Pick a different color that also contrasts with background
                other_colors = [c for c in color_names if c != target_color]
                cname = rng.choice(other_colors)
                rgb = _pick_shade(rng, cname)

            shapes.append(ShapeInstance(stype, cname, cx, cy, s, rgb_override=rgb))

        target_shape = shapes[target_idx]

        # Render with varied background
        img = render_image(shapes, canvas_size=(CANVAS_W, CANVAS_H), bg_color=bg_color)
        img_filename = f"{sample_id}.png"
        img.save(img_dir / img_filename)

        # Prompt
        desc = target_shape.shape_type.value
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