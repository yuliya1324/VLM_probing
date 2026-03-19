#src/data_preprocessing/vrd.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VRD_CSV_DIR = PROJECT_ROOT / "data" / "processed" / "vrd" / "csv"


TASK_TO_CSV = {
    "spatial": VRD_CSV_DIR / "vrd_spatial.csv",
    "color": VRD_CSV_DIR / "vrd_color.csv",
    "shape": VRD_CSV_DIR / "vrd_shape.csv",
}


SPATIAL_PROMPT = (
    "Determine the spatial relationship of '{subj}' relative to '{obj}'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below]\n"
    "Respond with ONLY the label. No explanation."
)

COLOR_PROMPT = (
    "What is the color of the {subj} in the image?\n"
    "Respond with ONLY the color name. No explanation."
)

SHAPE_PROMPT = (
    "What is the shape of the {subj} in the image?\n"
    "Choose ONE label from:\n"
    "[circular, oval, square, rectangular, triangular]\n"
    "Respond with ONLY the label. No explanation."
)

PROMPT_TEMPLATES = {
    "spatial": SPATIAL_PROMPT,
    "color": COLOR_PROMPT,
    "shape": SHAPE_PROMPT,
}


def load_vrd_dataframe(task: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load a VRD task CSV as a pandas DataFrame."""
    if task not in TASK_TO_CSV:
        raise ValueError(f"Unsupported task: {task}")

    path = Path(csv_path) if csv_path is not None else TASK_TO_CSV[task]
    return pd.read_csv(path)


def get_image_path(row: pd.Series) -> Path:
    """Return the image path stored in a VRD row."""
    if "image_path" in row.index:
        return Path(row["image_path"])
    if "img_path" in row.index:
        return Path(row["img_path"])
    raise ValueError("CSV must contain either 'image_path' or 'img_path'")


def build_prompt(row: pd.Series, task: str) -> str:
    """Build a task-specific prompt from a VRD row."""
    template = PROMPT_TEMPLATES[task]

    if task == "spatial":
        return template.format(
            subj=str(row["subj"]),
            obj=str(row["obj"]),
        )
    if task == "color":
        return template.format(
            subj=str(row["obj"]),
        )
    if task == "shape":
        return template.format(
            subj=str(row["obj"]),
        )

    raise ValueError(f"Unsupported task: {task}")


def get_label(row: pd.Series, task: str) -> str:
    """Extract the normalized ground-truth label from a VRD row."""
    if task == "spatial":
        return str(row["spatial"]).strip().lower()
    if task == "color":
        return str(row["color"]).strip().lower()
    if task == "shape":
        return str(row["shape"]).strip().lower()

    raise ValueError(f"Unsupported task: {task}")


def get_sample_id(row: pd.Series, task: str) -> str:
    """Build a unique sample identifier for resume / dedup use."""
    image_path = str(get_image_path(row))

    if task == "spatial":
        return f"{image_path}||{row['subj']}||{row['obj']}||{row['spatial']}"
    if task == "color":
        return f"{image_path}||{row['obj']}||{row['color']}"
    if task == "shape":
        return f"{image_path}||{row['obj']}||{row['color']}"

    raise ValueError(f"Unsupported task: {task}")

def resize_for_vrd(image: Image.Image, max_size: int = 448) -> Image.Image:
    """Resize image for VRD while preserving aspect ratio."""
    image = image.convert("RGB")
    return ImageOps.contain(image, (max_size, max_size))