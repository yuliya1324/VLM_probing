"""Data schemas for the synthetic datasets."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class ShapeType(Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    PENTAGON = "pentagon"
    STAR = "star"
    OVAL = "oval"
    RECTANGLE = "rectangle"


# Shape labels for the shape-identification task.
# Maps each ShapeType to the label the probe should predict.
# "circle" and "oval" are both "round" / "circular" / "oval" — but to avoid
# ambiguity we use distinct, non-overlapping labels:
SHAPE_LABELS = {
    ShapeType.CIRCLE: "circular",
    ShapeType.OVAL: "oval",
    ShapeType.SQUARE: "square",
    ShapeType.RECTANGLE: "rectangular",
    ShapeType.TRIANGLE: "triangular",
    # These are not used in the shape task (ambiguous):
    # ShapeType.PENTAGON: excluded
    # ShapeType.STAR: excluded
}

# Shapes used in the shape-identification task (unambiguous label set)
SHAPE_TASK_TYPES = [
    ShapeType.CIRCLE,
    ShapeType.OVAL,
    ShapeType.SQUARE,
    ShapeType.RECTANGLE,
    ShapeType.TRIANGLE,
]


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Base colors — one canonical RGB per color name
COLOR_BASE: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (30, 30, 30),
    "blue": (50, 80, 220),
    "brown": (140, 90, 40),
    "green": (50, 180, 50),
    "red": (220, 50, 50),
    "yellow": (230, 210, 40),
    "gray": (140, 140, 140),
    "orange": (240, 150, 30),
    "pink": (240, 110, 170),
    "purple": (150, 50, 200),
}

# Shade variations per color — the probe should generalize across these.
# Each color has 3-4 shades; the label stays the same.
COLOR_SHADES: dict[str, list[tuple[int, int, int]]] = {
    "white": [(255, 255, 255), (245, 245, 245), (235, 235, 240)],
    "black": [(30, 30, 30), (10, 10, 10), (50, 50, 50), (40, 35, 45)],
    "blue": [(50, 80, 220), (30, 60, 180), (70, 100, 240), (80, 130, 200)],
    "brown": [(140, 90, 40), (120, 70, 30), (160, 110, 55), (100, 65, 25)],
    "green": [(50, 180, 50), (30, 150, 30), (70, 200, 70), (40, 160, 80)],
    "red": [(220, 50, 50), (180, 30, 30), (240, 70, 60), (200, 40, 40)],
    "yellow": [(230, 210, 40), (210, 190, 30), (240, 230, 60), (220, 200, 20)],
    "gray": [(140, 140, 140), (110, 110, 110), (170, 170, 170), (155, 150, 160)],
    "orange": [(240, 150, 30), (220, 130, 20), (250, 170, 50), (230, 140, 40)],
    "pink": [(240, 110, 170), (220, 90, 150), (250, 140, 190), (230, 100, 160)],
    "purple": [(150, 50, 200), (130, 30, 180), (170, 70, 220), (140, 50, 170)],
}

# Legacy alias used by spatial.py and renderer
SHAPE_COLORS = COLOR_BASE

# Background colors to vary across samples
BACKGROUND_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 255),   # white
    (240, 240, 240),   # light gray
    (220, 220, 230),   # cool gray
    (245, 240, 230),   # warm cream
    (230, 240, 250),   # light blue
    (240, 235, 220),   # beige
    (200, 200, 200),   # medium gray
]


# ---------------------------------------------------------------------------
# Spatial relations
# ---------------------------------------------------------------------------

class SpatialRelation(Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"


# ---------------------------------------------------------------------------
# Individual shape on the canvas
# ---------------------------------------------------------------------------

@dataclass
class ShapeInstance:
    shape_type: ShapeType
    color_name: str          # key into COLOR_BASE
    cx: float                # center x  (pixel coords)
    cy: float                # center y
    size: float              # radius or half-side length
    rgb_override: Optional[tuple] = None  # if set, use this RGB instead of COLOR_BASE

    @property
    def rgb(self) -> tuple[int, int, int]:
        if self.rgb_override is not None:
            return self.rgb_override
        return COLOR_BASE[self.color_name]

    @property
    def descriptor(self) -> str:
        """Human-readable descriptor, e.g. 'red circle'."""
        return f"{self.color_name} {self.shape_type.value}"


# ---------------------------------------------------------------------------
# Dataset samples
# ---------------------------------------------------------------------------

@dataclass
class SpatialSample:
    """One sample for the spatial-relation probing dataset."""
    image_id: str
    image_filename: str
    subject: ShapeInstance
    reference: ShapeInstance
    relation: SpatialRelation
    prompt: str

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "subject_shape": self.subject.shape_type.value,
            "subject_color": self.subject.color_name,
            "reference_shape": self.reference.shape_type.value,
            "reference_color": self.reference.color_name,
            "relation": self.relation.value,
            "prompt": self.prompt,
        }


@dataclass
class ColorSample:
    """One sample for the color-identification probing dataset."""
    image_id: str
    image_filename: str
    target_shape: ShapeInstance
    color_label: str
    prompt: str

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "shape_type": self.target_shape.shape_type.value,
            "color_label": self.color_label,
            "prompt": self.prompt,
        }


@dataclass
class ShapeSample:
    """One sample for the shape-identification probing dataset."""
    image_id: str
    image_filename: str
    target_shape: ShapeInstance
    shape_label: str           # e.g. "circular", "rectangular", "triangular"
    prompt: str

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "color_name": self.target_shape.color_name,
            "shape_label": self.shape_label,
            "prompt": self.prompt,
        }