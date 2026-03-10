import json
import csv
from pathlib import Path
import re

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANN_PATH = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_annotations.json"
IMG_DIR  = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_images"

OUT_CSV  = PROJECT_ROOT / "data" / "vrd_attributes.csv"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s+\d+$")

def _base_label(name: str) -> str:
    name = (name or "").strip()
    m = _SUFFIX_RE.match(name)
    return m.group("base").strip() if m else name

def _load_annotations(ann_path: Path):
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _obj_name(obj: dict) -> str:
    names = obj.get("names", [])
    if names:
        return names[0]
    return "unknown"

def _extract_attributes(obj: dict):
    """
    object["attributes"] format example:
    [
        {"text": [...], "attribute": "gray"},
        {"text": [...], "attribute": "small"},
        ...
    ]
    """
    attrs = []
    for a in obj.get("attributes", []):
        attr = a.get("attribute", "")
        if attr:
            attrs.append(attr.strip())
    return attrs


# --------------------------------------------------
# Main CSV creation
# --------------------------------------------------
def build_attribute_csv(
    remove_object_number_suffix=False,
    skip_objects_without_attributes=True,
    max_attributes=None,
):
    if not ANN_PATH.exists():
        raise FileNotFoundError(ANN_PATH)
    if not IMG_DIR.exists():
        raise FileNotFoundError(IMG_DIR)

    data = _load_annotations(ANN_PATH)
    rows = []
    max_attr_len_found = 0

    for ex in data:
        # depending on your dataset, filename may exist;
        # if not, construct from photo_id
        if "filename" in ex:
            img_path = IMG_DIR / ex["filename"]
        else:
            # adapt extension if needed
            img_path = IMG_DIR / f'{ex["photo_id"]}.jpg'

        if not img_path.exists():
            continue

        for obj in ex.get("objects", []):
            obj_name = _obj_name(obj)
            if remove_object_number_suffix:
                obj_name = _base_label(obj_name)

            attrs = _extract_attributes(obj)

            if skip_objects_without_attributes and len(attrs) == 0:
                continue

            if max_attributes is not None:
                attrs = attrs[:max_attributes]

            max_attr_len_found = max(max_attr_len_found, len(attrs))

            row = {
                "image_path": str(img_path),
                "obj": obj_name,
            }

            for i, attr in enumerate(attrs, start=1):
                row[f"attribution{i}"] = attr

            rows.append(row)

    # decide header size
    if max_attributes is not None:
        num_attr_cols = max_attributes
    else:
        num_attr_cols = max_attr_len_found

    fieldnames = ["image_path", "obj"] + [
        f"attribution{i}" for i in range(1, num_attr_cols + 1)
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            # fill missing columns with empty string
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Total rows: {len(rows)}")
    print(f"Max number of attributes in one object: {num_attr_cols}")


if __name__ == "__main__":
    build_attribute_csv(
        remove_object_number_suffix=False,
        skip_objects_without_attributes=True,
        max_attributes=None,
    )