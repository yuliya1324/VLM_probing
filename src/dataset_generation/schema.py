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


# Named colors with distinct, easily distinguishable RGB values
SHAPE_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 50),
    "blue": (50, 80, 220),
    "green": (50, 180, 50),
    "yellow": (230, 210, 40),
    "purple": (150, 50, 200),
    "orange": (240, 150, 30),
    "pink": (240, 110, 170),
    "cyan": (40, 200, 210),
    "brown": (140, 90, 40),
    "gray": (140, 140, 140),
}


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
    color_name: str          # key into SHAPE_COLORS
    cx: float                # center x  (pixel coords)
    cy: float                # center y
    size: float              # radius or half-side length

    @property
    def rgb(self) -> tuple[int, int, int]:
        return SHAPE_COLORS[self.color_name]

    @property
    def descriptor(self) -> str:
        """Human-readable descriptor, e.g. 'red circle'."""
        return f"{self.color_name} {self.shape_type.value}"


# ---------------------------------------------------------------------------
# A single dataset sample
# ---------------------------------------------------------------------------

@dataclass
class SpatialSample:
    """One sample for the spatial-relation probing dataset."""
    image_id: str
    image_filename: str
    subject: ShapeInstance          # "the <subject> is <relation> the <reference>"
    reference: ShapeInstance
    relation: SpatialRelation
    prompt: str                    # text prompt fed to VLM

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
    """One sample for the color-identification probing dataset (sanity check)."""
    image_id: str
    image_filename: str
    target_shape: ShapeInstance
    color_label: str               # ground-truth color name
    prompt: str

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "shape_type": self.target_shape.shape_type.value,
            "color_label": self.color_label,
            "prompt": self.prompt,
        }