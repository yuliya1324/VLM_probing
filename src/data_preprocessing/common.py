#src/data_preprocessing/common.py

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANN_PATH = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_annotations.json"
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_images"
VRD_CSV_DIR = PROJECT_ROOT / "data" / "processed" / "vrd" / "csv"

_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s+\d+$")


def base_label(name: str) -> str:
    name = (name or "").strip()
    m = _SUFFIX_RE.match(name)
    return m.group("base").strip() if m else name


def load_annotations(ann_path: Path = ANN_PATH):
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)