"""Generate synthetic spatial-relationship dataset.

Each image contains exactly two shapes. The ground-truth relation describes
the position of the *subject* relative to the *reference*:
    "The <subject descriptor> is <relation> the <reference descriptor>."

Design choices to avoid shortcuts:
  - Shape types and colors are sampled randomly (no fixed pairings).
  - Sizes vary within a range.
  - Relations are enforced via placement constraints, then verified.
  - Each relation class is generated with roughly equal frequency.
  - Subject/reference roles are assigned independently of position.
"""

import json
import os
import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .schema import (
    SHAPE_COLORS,
    ShapeInstance,
    ShapeType,
    SpatialRelation,
    SpatialSample,
)
from .renderer import render_image

# Canvas parameters
CANVAS_W = 448
CANVAS_H = 448
MARGIN = 60          # keep shapes away from edges
MIN_GAP = 80         # minimum gap between shape centers on the diagnostic axis
SIZE_RANGE = (20, 40)  # half-side / radius range


def _random_shape(rng: random.Random) -> ShapeType:
    return rng.choice(list(ShapeType))


def _random_color_pair(rng: random.Random) -> tuple[str, str]:
    """Pick two distinct colors."""
    colors = list(SHAPE_COLORS.keys())
    c1, c2 = rng.sample(colors, 2)
    return c1, c2


def _place_pair(
    relation: SpatialRelation,
    rng: random.Random,
) -> tuple:
    """Return (subject_center, reference_center) satisfying the relation.

    We ensure a clear spatial gap along the diagnostic axis AND constrain
    the non-diagnostic axis so shapes are roughly aligned. This prevents
    diagonal placements where the label would be ambiguous.

    Alignment jitter: up to ±30px on the non-diagnostic axis so images
    aren't perfectly grid-like, but small enough that the relation is clear.
    """
    s_size = rng.uniform(*SIZE_RANGE)
    r_size = rng.uniform(*SIZE_RANGE)
    max_r = max(s_size, r_size)

    lo = MARGIN + max_r
    hi_x = CANVAS_W - MARGIN - max_r
    hi_y = CANVAS_H - MARGIN - max_r

    ALIGN_JITTER = 30  # max offset on the non-diagnostic axis

    if relation == SpatialRelation.LEFT_OF:
        # subject.x < reference.x, Y roughly aligned
        ref_x = rng.uniform(lo + MIN_GAP + max_r, hi_x)
        sub_x = rng.uniform(lo, ref_x - MIN_GAP)
        base_y = rng.uniform(lo, hi_y)
        sub_y = base_y + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        ref_y = base_y + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        sub_y = max(lo, min(hi_y, sub_y))
        ref_y = max(lo, min(hi_y, ref_y))

    elif relation == SpatialRelation.RIGHT_OF:
        # subject.x > reference.x, Y roughly aligned
        ref_x = rng.uniform(lo, hi_x - MIN_GAP - max_r)
        sub_x = rng.uniform(ref_x + MIN_GAP, hi_x)
        base_y = rng.uniform(lo, hi_y)
        sub_y = base_y + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        ref_y = base_y + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        sub_y = max(lo, min(hi_y, sub_y))
        ref_y = max(lo, min(hi_y, ref_y))

    elif relation == SpatialRelation.ABOVE:
        # subject.y < reference.y, X roughly aligned
        ref_y = rng.uniform(lo + MIN_GAP + max_r, hi_y)
        sub_y = rng.uniform(lo, ref_y - MIN_GAP)
        base_x = rng.uniform(lo, hi_x)
        sub_x = base_x + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        ref_x = base_x + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        sub_x = max(lo, min(hi_x, sub_x))
        ref_x = max(lo, min(hi_x, ref_x))

    elif relation == SpatialRelation.BELOW:
        # subject.y > reference.y, X roughly aligned
        ref_y = rng.uniform(lo, hi_y - MIN_GAP - max_r)
        sub_y = rng.uniform(ref_y + MIN_GAP, hi_y)
        base_x = rng.uniform(lo, hi_x)
        sub_x = base_x + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        ref_x = base_x + rng.uniform(-ALIGN_JITTER, ALIGN_JITTER)
        sub_x = max(lo, min(hi_x, sub_x))
        ref_x = max(lo, min(hi_x, ref_x))

    return (sub_x, sub_y, s_size), (ref_x, ref_y, r_size)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "The spatial relationship of {subject} to {reference} is",
    "Considering the image, the {subject} is positioned ___ the {reference}. The answer is",
    "Where is the {subject} relative to the {reference}? The {subject} is",
]


def _make_prompt(subject_desc: str, reference_desc: str, rng: random.Random) -> str:
    template = rng.choice(PROMPT_TEMPLATES)
    return template.format(subject=subject_desc, reference=reference_desc)


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_spatial_dataset(
    n_samples: int = 3000,
    output_dir: str = "data/raw/spatial",
    seed: int = 42,
    prompt_template_index: Optional[int] = None,  # None = random
) -> list[dict]:
    """Generate n_samples spatial-relation images and metadata.

    Returns list of sample dicts and writes:
      - <output_dir>/images/<id>.png
      - <output_dir>/metadata.json
    """
    rng = random.Random(seed)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    relations = list(SpatialRelation)
    samples_per_relation = n_samples // len(relations)
    # Build a balanced list of relations
    relation_list = []
    for rel in relations:
        relation_list.extend([rel] * samples_per_relation)
    # Fill remainder
    while len(relation_list) < n_samples:
        relation_list.append(rng.choice(relations))
    rng.shuffle(relation_list)

    samples: list[dict] = []

    for i, relation in enumerate(tqdm(relation_list, desc="Generating spatial dataset")):
        sample_id = f"spatial_{i:05d}"

        # Random shapes and colors
        s_type = _random_shape(rng)
        r_type = _random_shape(rng)
        s_color, r_color = _random_color_pair(rng)

        # Place shapes
        (sx, sy, s_size), (rx, ry, r_size) = _place_pair(relation, rng)

        subject = ShapeInstance(s_type, s_color, sx, sy, s_size)
        reference = ShapeInstance(r_type, r_color, rx, ry, r_size)

        # Render
        img = render_image([subject, reference], canvas_size=(CANVAS_W, CANVAS_H))
        img_filename = f"{sample_id}.png"
        img.save(img_dir / img_filename)

        # Prompt
        if prompt_template_index is not None:
            prompt = PROMPT_TEMPLATES[prompt_template_index].format(
                subject=subject.descriptor, reference=reference.descriptor
            )
        else:
            prompt = _make_prompt(subject.descriptor, reference.descriptor, rng)

        sample = SpatialSample(
            image_id=sample_id,
            image_filename=img_filename,
            subject=subject,
            reference=reference,
            relation=relation,
            prompt=prompt,
        )
        samples.append(sample.to_dict())

    # Save metadata
    meta_path = Path(output_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} spatial samples → {output_dir}")
    return samples