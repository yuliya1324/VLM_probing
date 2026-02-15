"""Render geometric shapes onto PIL images."""

import math
from PIL import Image, ImageDraw

from .schema import ShapeInstance, ShapeType


def _regular_polygon_points(cx: float, cy: float, r: float, n: int, rotation: float = 0) -> list[tuple[float, float]]:
    """Return vertices of a regular n-gon centered at (cx, cy) with circumradius r."""
    points = []
    for i in range(n):
        angle = rotation + 2 * math.pi * i / n
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def _star_points(cx: float, cy: float, r_outer: float, n: int = 5, ratio: float = 0.4) -> list[tuple[float, float]]:
    """Return vertices of a star with n outer points."""
    r_inner = r_outer * ratio
    points = []
    for i in range(2 * n):
        r = r_outer if i % 2 == 0 else r_inner
        angle = -math.pi / 2 + math.pi * i / n
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def draw_shape(draw: ImageDraw.ImageDraw, shape: ShapeInstance) -> None:
    """Draw a single ShapeInstance onto an ImageDraw canvas."""
    cx, cy, s = shape.cx, shape.cy, shape.size
    fill = shape.rgb

    if shape.shape_type == ShapeType.CIRCLE:
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=fill)

    elif shape.shape_type == ShapeType.SQUARE:
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=fill)

    elif shape.shape_type == ShapeType.TRIANGLE:
        pts = _regular_polygon_points(cx, cy, s, 3, rotation=-math.pi / 2)
        draw.polygon(pts, fill=fill)

    elif shape.shape_type == ShapeType.PENTAGON:
        pts = _regular_polygon_points(cx, cy, s, 5, rotation=-math.pi / 2)
        draw.polygon(pts, fill=fill)

    elif shape.shape_type == ShapeType.STAR:
        pts = _star_points(cx, cy, s)
        draw.polygon(pts, fill=fill)


def render_image(
    shapes: list[ShapeInstance],
    canvas_size: tuple[int, int] = (448, 448),
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Render a list of shapes onto a blank canvas and return the PIL Image."""
    img = Image.new("RGB", canvas_size, bg_color)
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        draw_shape(draw, shape)
    return img