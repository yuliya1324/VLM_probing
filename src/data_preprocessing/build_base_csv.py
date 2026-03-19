#src/data_preprocessing/build_base_csv.py

"""
Build base csv from VRD dataset.
"""

import csv

from common import ANN_PATH, IMG_DIR, VRD_CSV_DIR, load_annotations, base_label


OUT_CSV = VRD_CSV_DIR / "vrd_base.csv"


def obj_name(obj: dict) -> str:
    names = obj.get("names", [])
    return names[0] if names else "unknown"


def extract_attributes(obj: dict):
    attrs = []
    for a in obj.get("attributes", []):
        attr = a.get("attribute", "")
        if attr:
            attrs.append(attr.strip())
    return attrs


def bbox_area(obj: dict):
    bbox = obj.get("bbox", {})
    w = bbox.get("w")
    h = bbox.get("h")
    if w is None or h is None:
        return ""
    return w * h


def build_base_csv(
    remove_object_number_suffix=False,
    skip_objects_without_attributes=True,
    max_attributes=None,
):
    if not ANN_PATH.exists():
        raise FileNotFoundError(ANN_PATH)
    if not IMG_DIR.exists():
        raise FileNotFoundError(IMG_DIR)

    data = load_annotations()
    rows = []
    max_attr_len_found = 0

    for ex in data:
        if "filename" in ex:
            image_path = IMG_DIR / ex["filename"]
        else:
            image_path = IMG_DIR / f'{ex["photo_id"]}.jpg'

        if not image_path.exists():
            continue

        for obj in ex.get("objects", []):
            name = obj_name(obj)
            if remove_object_number_suffix:
                name = base_label(name)

            attrs = extract_attributes(obj)

            if skip_objects_without_attributes and not attrs:
                continue

            if max_attributes is not None:
                attrs = attrs[:max_attributes]

            max_attr_len_found = max(max_attr_len_found, len(attrs))

            row = {
                "image_path": str(image_path),
                "obj": name,
                "area": bbox_area(obj),
            }

            for i, attr in enumerate(attrs, start=1):
                row[f"attribution{i}"] = attr

            rows.append(row)

    num_attr_cols = max_attributes if max_attributes is not None else max_attr_len_found

    fieldnames = ["image_path", "obj", "area"] + [
        f"attribution{i}" for i in range(1, num_attr_cols + 1)
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Total rows: {len(rows)}")
    print(f"Max number of attributes in one object: {num_attr_cols}")


if __name__ == "__main__":
    build_base_csv(
        remove_object_number_suffix=False,
        skip_objects_without_attributes=True,
        max_attributes=None,
    )